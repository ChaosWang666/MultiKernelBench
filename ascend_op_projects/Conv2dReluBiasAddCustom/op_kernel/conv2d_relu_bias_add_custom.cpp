
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelConv2dReluBiasAdd {
public:
    __aicore__ inline KernelConv2dReluBiasAdd() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR bias, GM_ADDR z,
                                 uint32_t totalLength, uint32_t tileNum,
                                 uint32_t outChannels, uint32_t spatialSize)
    {
        this->totalLength = totalLength;
        this->outChannels = outChannels;
        this->spatialSize = spatialSize;
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ float *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        biasGm.SetGlobalBuffer((__gm__ float *)bias, outChannels);
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
        AscendC::LocalTensor<float> biasLocal = tmpBuf.Get<float>();

        // Apply ReLU: zLocal = max(xLocal, 0)
        float zeroVal = 0.0f;
        AscendC::Maxs(zLocal, xLocal, zeroVal, this->tileLength);

        // Compute global offset for this tile
        uint32_t globalOffset = this->blockLength * AscendC::GetBlockIdx() + progress * this->tileLength;

        // Add bias: for each element, determine its channel and add bias[channel]
        // x is (N, C, H, W), so element at global index i has channel = (i / spatialSize) % outChannels
        for (uint32_t j = 0; j < this->tileLength; j++) {
            uint32_t gIdx = globalOffset + j;
            uint32_t channel = (gIdx / this->spatialSize) % this->outChannels;
            biasLocal.SetValue(j, biasGm.GetValue(channel));
        }

        AscendC::Add(zLocal, zLocal, biasLocal, this->tileLength);

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
    AscendC::GlobalTensor<float> zGm;
    uint32_t totalLength;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t outChannels;
    uint32_t spatialSize;
};

extern "C" __global__ __aicore__ void conv2d_relu_bias_add_custom(GM_ADDR x, GM_ADDR bias, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelConv2dReluBiasAdd op;
    op.Init(x, bias, z, tiling_data.totalLength, tiling_data.tileNum, tiling_data.outChannels, tiling_data.spatialSize);
    op.Process();
}
