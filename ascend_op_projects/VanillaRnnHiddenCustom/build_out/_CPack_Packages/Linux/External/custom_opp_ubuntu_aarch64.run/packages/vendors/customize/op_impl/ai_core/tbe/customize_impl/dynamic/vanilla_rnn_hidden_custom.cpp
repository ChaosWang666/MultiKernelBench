
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue
 
class KernelVanillaRnnHidden {
public:
    __aicore__ inline KernelVanillaRnnHidden() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR hidden, GM_ADDR weight_ih, GM_ADDR bias_ih, GM_ADDR hidden_new, uint32_t totalLength, uint32_t tileNum)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        inputGm.SetGlobalBuffer((__gm__ DTYPE_INPUT *)input + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        hiddenGm.SetGlobalBuffer((__gm__ DTYPE_HIDDEN *)hidden + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        weightIhGm.SetGlobalBuffer((__gm__ DTYPE_WEIGHT_IH *)weight_ih, this->blockLength * this->blockLength);
        biasIhGm.SetGlobalBuffer((__gm__ DTYPE_BIAS_IH *)bias_ih, this->blockLength);
        hiddenNewGm.SetGlobalBuffer((__gm__ DTYPE_HIDDEN_NEW *)hidden_new + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        pipe.InitBuffer(inQueueInput, BUFFER_NUM, this->tileLength * sizeof(DTYPE_INPUT));
        pipe.InitBuffer(inQueueHidden, BUFFER_NUM, this->tileLength * sizeof(DTYPE_HIDDEN));
        pipe.InitBuffer(inQueueWeightIh, BUFFER_NUM, this->tileLength * this->tileLength * sizeof(DTYPE_WEIGHT_IH));
        pipe.InitBuffer(inQueueBiasIh, BUFFER_NUM, this->tileLength * sizeof(DTYPE_BIAS_IH));
        pipe.InitBuffer(outQueueHiddenNew, BUFFER_NUM, this->tileLength * sizeof(DTYPE_HIDDEN_NEW));
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
        AscendC::LocalTensor<DTYPE_INPUT> inputLocal = inQueueInput.AllocTensor<DTYPE_INPUT>();
        AscendC::LocalTensor<DTYPE_HIDDEN> hiddenLocal = inQueueHidden.AllocTensor<DTYPE_HIDDEN>();
        AscendC::LocalTensor<DTYPE_WEIGHT_IH> weightIhLocal = inQueueWeightIh.AllocTensor<DTYPE_WEIGHT_IH>();
        AscendC::LocalTensor<DTYPE_BIAS_IH> biasIhLocal = inQueueBiasIh.AllocTensor<DTYPE_BIAS_IH>();
        AscendC::DataCopy(inputLocal, inputGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(hiddenLocal, hiddenGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(weightIhLocal, weightIhGm[progress * this->tileLength], this->tileLength * this->tileLength);
        AscendC::DataCopy(biasIhLocal, biasIhGm[progress * this->tileLength], this->tileLength);
        inQueueInput.EnQue(inputLocal);
        inQueueHidden.EnQue(hiddenLocal);
        inQueueWeightIh.EnQue(weightIhLocal);
        inQueueBiasIh.EnQue(biasIhLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_INPUT> inputLocal = inQueueInput.DeQue<DTYPE_INPUT>();
        AscendC::LocalTensor<DTYPE_HIDDEN> hiddenLocal = inQueueHidden.DeQue<DTYPE_HIDDEN>();
        AscendC::LocalTensor<DTYPE_WEIGHT_IH> weightIhLocal = inQueueWeightIh.DeQue<DTYPE_WEIGHT_IH>();
        AscendC::LocalTensor<DTYPE_BIAS_IH> biasIhLocal = inQueueBiasIh.DeQue<DTYPE_BIAS_IH>();
        AscendC::LocalTensor<DTYPE_HIDDEN_NEW> hiddenNewLocal = outQueueHiddenNew.AllocTensor<DTYPE_HIDDEN_NEW>();
        // Matrix multiplication and addition
        AscendC::MatMul(hiddenNewLocal, inputLocal, hiddenLocal, weightIhLocal, biasIhLocal, this->tileLength, this->tileLength);
        outQueueHiddenNew.EnQue<DTYPE_HIDDEN_NEW>(hiddenNewLocal);
        inQueueInput.FreeTensor(inputLocal);
        inQueueHidden.FreeTensor(hiddenLocal);
        inQueueWeightIh.FreeTensor(weightIhLocal);
        inQueueBiasIh.FreeTensor(biasIhLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_HIDDEN_NEW> hiddenNewLocal = outQueueHiddenNew.DeQue<DTYPE_HIDDEN_NEW>();
        AscendC::DataCopy(hiddenNewGm[progress * this->tileLength], hiddenNewLocal, this->tileLength);
        outQueueHiddenNew.FreeTensor(hiddenNewLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueInput, inQueueHidden, inQueueWeightIh, inQueueBiasIh;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueHiddenNew;
    AscendC::GlobalTensor<DTYPE_INPUT> inputGm;
    AscendC::GlobalTensor<DTYPE_HIDDEN> hiddenGm;
    AscendC::GlobalTensor<DTYPE_WEIGHT_IH> weightIhGm;
    AscendC::GlobalTensor<DTYPE_BIAS_IH> biasIhGm;
    AscendC::GlobalTensor<DTYPE_HIDDEN_NEW> hiddenNewGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void vanilla_rnn_hidden_custom(GM_ADDR input, GM_ADDR hidden, GM_ADDR weight_ih, GM_ADDR bias_ih, GM_ADDR hidden_new, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelVanillaRnnHidden op;
    op.Init(input, hidden, weight_ih, bias_ih, hidden_new, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
