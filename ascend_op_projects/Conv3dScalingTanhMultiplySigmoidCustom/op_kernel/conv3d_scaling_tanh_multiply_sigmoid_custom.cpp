
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelConv3dScalingTanhMultiplySigmoidCustom {
public:
    __aicore__ inline KernelConv3dScalingTanhMultiplySigmoidCustom() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR scaling_factor, GM_ADDR bias, GM_ADDR z,
                                 uint32_t totalLength, uint32_t tileNum,
                                 uint32_t outChannels, uint32_t spatialSize)
    {
        this->outChannels = outChannels;
        this->spatialSize = spatialSize;
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ float *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        scalingGm.SetGlobalBuffer((__gm__ float *)scaling_factor, outChannels);
        biasGm.SetGlobalBuffer((__gm__ float *)bias, outChannels);
        zGm.SetGlobalBuffer((__gm__ float *)z + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmpBuf1, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmpBuf2, this->tileLength * sizeof(float));

        // Copy scaling_factor and bias to UB
        uint32_t alignedChannels = ((outChannels + 7) / 8) * 8;
        pipe.InitBuffer(scalingBuf, alignedChannels * sizeof(float));
        pipe.InitBuffer(biasBuf, alignedChannels * sizeof(float));

        AscendC::LocalTensor<float> scalingLocal = scalingBuf.Get<float>();
        AscendC::LocalTensor<float> biasLocal = biasBuf.Get<float>();
        AscendC::DataCopy(scalingLocal, scalingGm[0], alignedChannels);
        AscendC::DataCopy(biasLocal, biasGm[0], alignedChannels);
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
        AscendC::LocalTensor<float> scalingLocal = scalingBuf.Get<float>();
        AscendC::LocalTensor<float> biasLocal = biasBuf.Get<float>();
        AscendC::LocalTensor<float> temp1 = tmpBuf1.Get<float>();
        AscendC::LocalTensor<float> temp2 = tmpBuf2.Get<float>();

        // Compute global offset for this tile
        uint32_t globalOffset = this->blockLength * AscendC::GetBlockIdx() + progress * this->tileLength;

        // For each element, determine its channel: channel = (globalOffset / spatialSize) % outChannels
        // Then: z = sigmoid(tanh(x * scaling[ch]) * bias[ch])

        // Process element by element for correctness with broadcasting
        // Build scaling and bias vectors for the tile
        for (uint32_t i = 0; i < this->tileLength; i++) {
            uint32_t elemGlobalIdx = globalOffset + i;
            uint32_t ch = (elemGlobalIdx / this->spatialSize) % this->outChannels;
            temp1.SetValue(i, scalingLocal.GetValue(ch));
            temp2.SetValue(i, biasLocal.GetValue(ch));
        }

        // x * scaling_factor
        AscendC::Mul(zLocal, xLocal, temp1, this->tileLength);
        // tanh
        AscendC::Tanh(zLocal, zLocal, this->tileLength);
        // * bias
        AscendC::Mul(zLocal, zLocal, temp2, this->tileLength);
        // sigmoid: 1 / (1 + exp(-x))
        // Use AscendC::Sigmoid if available, otherwise manually compute
        AscendC::Sigmoid(zLocal, zLocal, this->tileLength);

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
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf1, tmpBuf2;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scalingBuf, biasBuf;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> scalingGm;
    AscendC::GlobalTensor<float> biasGm;
    AscendC::GlobalTensor<float> zGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t outChannels;
    uint32_t spatialSize;
};

extern "C" __global__ __aicore__ void conv3d_scaling_tanh_multiply_sigmoid_custom(GM_ADDR x, GM_ADDR scaling_factor, GM_ADDR bias, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelConv3dScalingTanhMultiplySigmoidCustom op;
    op.Init(x, scaling_factor, bias, z, tiling_data.totalLength, tiling_data.tileNum, tiling_data.outChannels, tiling_data.spatialSize);
    op.Process();
}
