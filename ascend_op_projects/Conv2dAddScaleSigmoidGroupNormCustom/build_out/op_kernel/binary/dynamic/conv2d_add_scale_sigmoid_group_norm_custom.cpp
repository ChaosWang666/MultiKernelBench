
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelAddScaleSigmoid {
public:
    __aicore__ inline KernelAddScaleSigmoid() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR bias, GM_ADDR scale, GM_ADDR z,
                                 uint32_t totalLength, uint32_t tileNum,
                                 uint32_t batchSize, uint32_t channels, uint32_t spatialSize)
    {
        this->totalLength = totalLength;
        this->channels = channels;
        this->spatialSize = spatialSize;
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        // Align tileLength to 32 bytes (8 floats)
        if (this->tileLength == 0) {
            this->tileLength = 8;
        }

        xGm.SetGlobalBuffer((__gm__ float *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        biasGm.SetGlobalBuffer((__gm__ float *)bias, channels);
        scaleGm.SetGlobalBuffer((__gm__ float *)scale, channels);
        zGm.SetGlobalBuffer((__gm__ float *)z + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmpBuf, 1, this->tileLength * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
        AscendC::LocalTensor<float> tmpLocal = tmpBuf.Get<float>();

        uint32_t globalOffset = this->blockLength * AscendC::GetBlockIdx() + progress * this->tileLength;

        // For each element, determine channel index: globalOffset / spatialSize gives (batch*channels + channel)
        // channel = (globalOffset / spatialSize) % channels
        // We process element by element for bias/scale broadcast
        for (uint32_t i = 0; i < this->tileLength; i++) {
            uint32_t globalIdx = globalOffset + i;
            uint32_t channelIdx = (globalIdx / this->spatialSize) % this->channels;
            float biasVal, scaleVal;
            // Read bias and scale for this channel
            // We'll use scalar reads
            biasVal = *(((__gm__ float*)biasGm.GetPhyAddr()) + channelIdx);
            scaleVal = *(((__gm__ float*)scaleGm.GetPhyAddr()) + channelIdx);
            float val = xLocal.GetValue(i);
            val = val + biasVal;
            val = val * scaleVal;
            // sigmoid: 1 / (1 + exp(-val))
            float expNeg;
            if (val > 0) {
                expNeg = 1.0f / (1.0f + AscendC::Exp(-val));
            } else {
                float ev = AscendC::Exp(val);
                expNeg = ev / (1.0f + ev);
            }
            zLocal.SetValue(i, expNeg);
        }

        outQueueZ.EnQue<float>(zLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<float> zLocal = outQueueZ.DeQue<float>();
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> biasGm;
    AscendC::GlobalTensor<float> scaleGm;
    AscendC::GlobalTensor<float> zGm;
    uint32_t totalLength;
    uint32_t channels;
    uint32_t spatialSize;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void conv2d_add_scale_sigmoid_group_norm_custom(GM_ADDR x, GM_ADDR bias, GM_ADDR scale, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelAddScaleSigmoid op;
    op.Init(x, bias, scale, z, tiling_data.totalLength, tiling_data.tileNum,
            tiling_data.batchSize, tiling_data.channels, tiling_data.spatialSize);
    op.Process();
}
