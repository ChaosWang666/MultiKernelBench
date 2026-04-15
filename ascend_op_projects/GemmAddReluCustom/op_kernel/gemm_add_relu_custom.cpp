
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelGemmAddRelu {
public:
    __aicore__ inline KernelGemmAddRelu() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR bias, GM_ADDR y, uint32_t batch, uint32_t inFeatures, uint32_t outFeatures)
    {
        this->batch = batch;
        this->inFeatures = inFeatures;
        this->outFeatures = outFeatures;
        this->blockLength = inFeatures * outFeatures / AscendC::GetBlockNum();
        this->tileNum = 4096;
        this->tileLength = this->blockLength / this->tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ float *)x, batch * inFeatures);
        wGm.SetGlobalBuffer((__gm__ float *)w, inFeatures * outFeatures);
        biasGm.SetGlobalBuffer((__gm__ float *)bias, outFeatures);
        yGm.SetGlobalBuffer((__gm__ float *)y, batch * outFeatures);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(inQueueW, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(inQueueBias, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(float));
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
        AscendC::LocalTensor<float> wLocal = inQueueW.AllocTensor<float>();
        AscendC::LocalTensor<float> biasLocal = inQueueBias.AllocTensor<float>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(wLocal, wGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(biasLocal, biasGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
        inQueueW.EnQue(wLocal);
        inQueueBias.EnQue(biasLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> wLocal = inQueueW.DeQue<float>();
        AscendC::LocalTensor<float> biasLocal = inQueueBias.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        AscendC::MatMul(yLocal, xLocal, wLocal, this->tileLength);
        AscendC::Add(yLocal, yLocal, biasLocal, this->tileLength);
        AscendC::ReLU(yLocal, yLocal, this->tileLength);
        outQueueY.EnQue<float>(yLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueW.FreeTensor(wLocal);
        inQueueBias.FreeTensor(biasLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<float> yLocal = outQueueY.DeQue<float>();
        AscendC::DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueW, inQueueBias;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> wGm;
    AscendC::GlobalTensor<float> biasGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t batch;
    uint32_t inFeatures;
    uint32_t outFeatures;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void gemm_add_relu_custom(GM_ADDR x, GM_ADDR w, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelGemmAddRelu op;
    op.Init(x, w, bias, y, tiling_data.batch, tiling_data.inFeatures, tiling_data.outFeatures);
    op.Process();
}
