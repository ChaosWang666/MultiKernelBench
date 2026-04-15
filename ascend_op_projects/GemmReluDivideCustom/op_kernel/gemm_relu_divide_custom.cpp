
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue
 
class KernelGemmReluDivide {
public:
    __aicore__ inline KernelGemmReluDivide() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR bias, GM_ADDR z, uint32_t batchSize, uint32_t inFeatures, uint32_t outFeatures, uint32_t tileNum)
    {
        this->batchSize = batchSize;
        this->inFeatures = inFeatures;
        this->outFeatures = outFeatures;
        this->tileNum = tileNum;
        this->blockLength = outFeatures / AscendC::GetBlockNum();
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x, batchSize * inFeatures);
        wGm.SetGlobalBuffer((__gm__ DTYPE_W *)w, inFeatures * outFeatures);
        biasGm.SetGlobalBuffer((__gm__ DTYPE_BIAS *)bias, outFeatures);
        zGm.SetGlobalBuffer((__gm__ DTYPE_Z *)z, batchSize * outFeatures);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(inQueueW, BUFFER_NUM, this->tileLength * sizeof(DTYPE_W));
        pipe.InitBuffer(inQueueBias, BUFFER_NUM, this->tileLength * sizeof(DTYPE_BIAS));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Z));
    }
    __aicore__ inline void Process()
    {
        for (uint32_t batch = 0; batch < batchSize; batch++) {
            for (int32_t i = 0; i < tileNum * BUFFER_NUM; i++) {
                CopyIn(i, batch);
                Compute(i, batch);
                CopyOut(i, batch);
            }
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t batch)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_W> wLocal = inQueueW.AllocTensor<DTYPE_W>();
        AscendC::LocalTensor<DTYPE_BIAS> biasLocal = inQueueBias.AllocTensor<DTYPE_BIAS>();
        AscendC::DataCopy(xLocal, xGm[batch * inFeatures], inFeatures);
        AscendC::DataCopy(wLocal, wGm[progress * this->tileLength], inFeatures);
        AscendC::DataCopy(biasLocal, biasGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
        inQueueW.EnQue(wLocal);
        inQueueBias.EnQue(biasLocal);
    }
    __aicore__ inline void Compute(int32_t progress, uint32_t batch)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_W> wLocal = inQueueW.DeQue<DTYPE_W>();
        AscendC::LocalTensor<DTYPE_BIAS> biasLocal = inQueueBias.DeQue<DTYPE_BIAS>();
        AscendC::LocalTensor<DTYPE_Z> zLocal = outQueueZ.AllocTensor<DTYPE_Z>();
        AscendC::MatMul(zLocal, xLocal, wLocal, biasLocal, inFeatures, this->tileLength);
        AscendC::ReLU(zLocal, zLocal, this->tileLength);
        AscendC::Div(zLocal, zLocal, 2.0f, this->tileLength); // Assuming divisor is 2.0
        outQueueZ.EnQue<DTYPE_Z>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueW.FreeTensor(wLocal);
        inQueueBias.FreeTensor(biasLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress, uint32_t batch)
    {
        AscendC::LocalTensor<DTYPE_Z> zLocal = outQueueZ.DeQue<DTYPE_Z>();
        AscendC::DataCopy(zGm[batch * outFeatures + progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueW, inQueueBias;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_W> wGm;
    AscendC::GlobalTensor<DTYPE_BIAS> biasGm;
    AscendC::GlobalTensor<DTYPE_Z> zGm;
    uint32_t batchSize;
    uint32_t inFeatures;
    uint32_t outFeatures;
    uint32_t tileNum;
    uint32_t blockLength;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void gemm_relu_divide_custom(GM_ADDR x, GM_ADDR w, GM_ADDR bias, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelGemmReluDivide op;
    op.Init(x, w, bias, z, tiling_data.batchSize, tiling_data.inFeatures, tiling_data.outFeatures, tiling_data.tileNum);
    op.Process();
}
