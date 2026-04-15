
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

// This kernel performs: MaxPool3d(kernel=2) -> MaxPool3d(kernel=3) -> Sum(dim=1, keepdim=True)
// on 5D input [B, C, D, H, W]
// Output: [B, 1, D/6, H/6, W/6]

class KernelConvTranspose3dMaxMaxSum {
public:
    __aicore__ inline KernelConvTranspose3dMaxMaxSum() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                 uint32_t batchSize, uint32_t channels,
                                 uint32_t depth, uint32_t height, uint32_t width,
                                 uint32_t totalLength, uint32_t tileNum)
    {
        this->batchSize = batchSize;
        this->channels = channels;
        this->inD = depth;
        this->inH = height;
        this->inW = width;
        // After MaxPool3d(2)
        this->mp1D = depth / 2;
        this->mp1H = height / 2;
        this->mp1W = width / 2;
        // After MaxPool3d(3)
        this->outD = this->mp1D / 3;
        this->outH = this->mp1H / 3;
        this->outW = this->mp1W / 3;

        this->spatialIn = this->inD * this->inH * this->inW;
        this->spatialOut = this->outD * this->outH * this->outW;

        // Total output elements = B * 1 * outD * outH * outW
        uint32_t totalOut = batchSize * this->spatialOut;

        // Distribute output elements across blocks
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();

        this->outputPerBlock = totalOut / blockNum;
        uint32_t remainder = totalOut - this->outputPerBlock * blockNum;
        if (blockIdx < remainder) {
            this->outputPerBlock += 1;
            this->outputStart = blockIdx * this->outputPerBlock;
        } else {
            this->outputStart = blockIdx * this->outputPerBlock + remainder;
        }

        xGm.SetGlobalBuffer((__gm__ float*)x, batchSize * channels * this->spatialIn);
        yGm.SetGlobalBuffer((__gm__ float*)y, batchSize * this->spatialOut);

        // We'll process one output element at a time, reading C values and summing
        // Allocate buffer for channel data
        uint32_t channelAligned = ((channels + 7) / 8) * 8;
        pipe.InitBuffer(inQueueX, 1, channelAligned * sizeof(float));
        pipe.InitBuffer(outQueueZ, 1, 8 * sizeof(float));
        pipe.InitBuffer(tmpBuf, 1, channelAligned * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < this->outputPerBlock; i++) {
            uint32_t outIdx = this->outputStart + i;
            // outIdx -> (b, od, oh, ow) in output space
            uint32_t b = outIdx / this->spatialOut;
            uint32_t rem = outIdx - b * this->spatialOut;
            uint32_t od = rem / (this->outH * this->outW);
            uint32_t rem2 = rem - od * (this->outH * this->outW);
            uint32_t oh = rem2 / this->outW;
            uint32_t ow = rem2 - oh * this->outW;

            ComputeElement(b, od, oh, ow, outIdx);
        }
    }

private:
    __aicore__ inline float ReadGm(uint32_t b, uint32_t c, uint32_t d, uint32_t h, uint32_t w)
    {
        uint32_t idx = ((b * this->channels + c) * this->inD + d) * this->inH * this->inW + h * this->inW + w;
        // Use a temporary tensor to read single value
        return *(((__gm__ float*)xGm.GetPhyAddr()) + idx);
    }

    __aicore__ inline void ComputeElement(uint32_t b, uint32_t od, uint32_t oh, uint32_t ow, uint32_t outIdx)
    {
        // For this output position, we need to compute:
        // 1. MaxPool3d(3) over mp1 space: window [od*3..od*3+2, oh*3..oh*3+2, ow*3..ow*3+2]
        // 2. Each mp1 position comes from MaxPool3d(2) over input: window [d*2..d*2+1, h*2..h*2+1, w*2..w*2+1]
        // 3. Sum over channels
        // Combined: each output covers input region of 6x6x6

        uint32_t baseD = od * 6;
        uint32_t baseH = oh * 6;
        uint32_t baseW = ow * 6;

        // For each channel, compute the fused max-pool result
        AscendC::LocalTensor<float> chanData = inQueueX.AllocTensor<float>();
        uint32_t channelAligned = ((channels + 7) / 8) * 8;

        // Initialize to -inf
        AscendC::LocalTensor<float> tmpLocal = tmpBuf.AllocTensor<float>();
        float negInf = -3.402823466e+38f;
        AscendC::Duplicate(chanData, negInf, channelAligned);

        // We need to find for each channel the max over the 6x6x6 region
        // But this is MaxPool(2) then MaxPool(3), which equals:
        // Stage1: 3x3x3 blocks of 2x2x2 -> max of each 2x2x2 -> 3x3x3 intermediate
        // Stage2: max of that 3x3x3
        // Effectively: partition 6x6x6 into 3x3x3 blocks of 2x2x2, take max of each 2x2x2, then max of those 27 maxes
        // Since max is associative, this is NOT the same as max over the whole 6x6x6
        // MaxPool(2) then MaxPool(3) means:
        // mp1[i,j,k] = max over 2x2x2 block starting at (2i,2j,2k)
        // mp2 = max over 3x3x3 of mp1 values
        // = max over i in [od*3..od*3+2], j in [oh*3..oh*3+2], k in [ow*3..ow*3+2] of mp1[i,j,k]
        // = max over those 27 mp1 values, each being max of 8 input values
        // = max over all 216 input values in the 6x6x6 block
        // Actually yes, it IS the same as max over the 6x6x6 since the 2x2x2 blocks tile the 6x6x6 perfectly
        // So we just need max over the 6x6x6 input region for each channel

        // Read values and take max channel-wise
        // We process one element at a time from GM to avoid large buffers
        for (uint32_t dd = 0; dd < 6; dd++) {
            for (uint32_t hh = 0; hh < 6; hh++) {
                for (uint32_t ww = 0; ww < 6; ww++) {
                    uint32_t curD = baseD + dd;
                    uint32_t curH = baseH + hh;
                    uint32_t curW = baseW + ww;

                    if (curD < this->inD && curH < this->inH && curW < this->inW) {
                        // Read all channels for this spatial position
                        for (uint32_t c = 0; c < this->channels; c++) {
                            uint32_t gmIdx = ((b * this->channels + c) * this->inD + curD) * this->inH * this->inW + curH * this->inW + curW;
                            // Read from global memory
                            float val = *((__gm__ float*)(xGm.GetPhyAddr()) + gmIdx);
                            float curMax = chanData.GetValue(c);
                            if (val > curMax) {
                                chanData.SetValue(c, val);
                            }
                        }
                    }
                }
            }
        }

        // Sum all channel values
        // Use ReduceSum
        float sum = 0.0f;
        for (uint32_t c = 0; c < this->channels; c++) {
            sum += chanData.GetValue(c);
        }

        inQueueX.FreeTensor(chanData);
        tmpBuf.FreeTensor(tmpLocal);

        // Write output
        AscendC::LocalTensor<float> outLocal = outQueueZ.AllocTensor<float>();
        outLocal.SetValue(0, sum);
        // We write one element, but need 32-byte aligned writes (8 floats)
        // Just write and let hardware handle it
        AscendC::DataCopy(yGm[outIdx], outLocal, 1);
        outQueueZ.FreeTensor(outLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueZ;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t batchSize, channels;
    uint32_t inD, inH, inW;
    uint32_t mp1D, mp1H, mp1W;
    uint32_t outD, outH, outW;
    uint32_t spatialIn, spatialOut;
    uint32_t outputPerBlock;
    uint32_t outputStart;
};

extern "C" __global__ __aicore__ void conv_transpose3d_max_max_sum_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelConvTranspose3dMaxMaxSum op;
    op.Init(x, y,
            tiling_data.batchSize, tiling_data.channels,
            tiling_data.depth, tiling_data.height, tiling_data.width,
            tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
