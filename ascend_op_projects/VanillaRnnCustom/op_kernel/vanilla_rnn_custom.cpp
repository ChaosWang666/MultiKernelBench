
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelVanillaRnn {
public:
    __aicore__ inline KernelVanillaRnn() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR hidden, GM_ADDR i2hWeight, GM_ADDR i2hBias, GM_ADDR h2oWeight, GM_ADDR h2oBias, GM_ADDR output, GM_ADDR newHidden,
                                uint32_t batchSize, uint32_t inputSize, uint32_t hiddenSize, uint32_t outputSize, uint32_t tileNum)
    {
        this->batchSize = batchSize;
        this->inputSize = inputSize;
        this->hiddenSize = hiddenSize;
        this->outputSize = outputSize;
        this->tileNum = tileNum;
        this->blockLength = batchSize * (inputSize + hiddenSize) / AscendC::GetBlockNum();
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x, this->blockLength);
        hiddenGm.SetGlobalBuffer((__gm__ DTYPE_HIDDEN *)hidden, this->blockLength);
        i2hWeightGm.SetGlobalBuffer((__gm__ DTYPE_I2H_WEIGHT *)i2hWeight, (inputSize + hiddenSize) * hiddenSize * sizeof(DTYPE_I2H_WEIGHT));
        i2hBiasGm.SetGlobalBuffer((__gm__ DTYPE_I2H_BIAS *)i2hBias, hiddenSize * sizeof(DTYPE_I2H_BIAS));
        h2oWeightGm.SetGlobalBuffer((__gm__ DTYPE_H2O_WEIGHT *)h2oWeight, hiddenSize * outputSize * sizeof(DTYPE_H2O_WEIGHT));
        h2oBiasGm.SetGlobalBuffer((__gm__ DTYPE_H2O_BIAS *)h2oBias, outputSize * sizeof(DTYPE_H2O_BIAS));
        outputGm.SetGlobalBuffer((__gm__ DTYPE_OUTPUT *)output, batchSize * outputSize * sizeof(DTYPE_OUTPUT));
        newHiddenGm.SetGlobalBuffer((__gm__ DTYPE_NEW_HIDDEN *)newHidden, batchSize * hiddenSize * sizeof(DTYPE_NEW_HIDDEN));

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(inQueueHidden, BUFFER_NUM, this->tileLength * sizeof(DTYPE_HIDDEN));
        pipe.InitBuffer(inQueueI2hWeight, BUFFER_NUM, (inputSize + hiddenSize) * hiddenSize * sizeof(DTYPE_I2H_WEIGHT));
        pipe.InitBuffer(inQueueI2hBias, BUFFER_NUM, hiddenSize * sizeof(DTYPE_I2H_BIAS));
        pipe.InitBuffer(inQueueH2oWeight, BUFFER_NUM, hiddenSize * outputSize * sizeof(DTYPE_H2O_WEIGHT));
        pipe.InitBuffer(inQueueH2oBias, BUFFER_NUM, outputSize * sizeof(DTYPE_H2O_BIAS));
        pipe.InitBuffer(outQueueOutput, BUFFER_NUM, batchSize * outputSize * sizeof(DTYPE_OUTPUT));
        pipe.InitBuffer(outQueueNewHidden, BUFFER_NUM, batchSize * hiddenSize * sizeof(DTYPE_NEW_HIDDEN));
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
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_HIDDEN> hiddenLocal = inQueueHidden.AllocTensor<DTYPE_HIDDEN>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(hiddenLocal, hiddenGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
        inQueueHidden.EnQue(hiddenLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_HIDDEN> hiddenLocal = inQueueHidden.DeQue<DTYPE_HIDDEN>();
        AscendC::LocalTensor<DTYPE_I2H_WEIGHT> i2hWeightLocal = inQueueI2hWeight.AllocTensor<DTYPE_I2H_WEIGHT>();
        AscendC::LocalTensor<DTYPE_I2H_BIAS> i2hBiasLocal = inQueueI2hBias.AllocTensor<DTYPE_I2H_BIAS>();
        AscendC::LocalTensor<DTYPE_H2O_WEIGHT> h2oWeightLocal = inQueueH2oWeight.AllocTensor<DTYPE_H2O_WEIGHT>();
        AscendC::LocalTensor<DTYPE_H2O_BIAS> h2oBiasLocal = inQueueH2oBias.AllocTensor<DTYPE_H2O_BIAS>();
        AscendC::LocalTensor<DTYPE_NEW_HIDDEN> newHiddenLocal = outQueueNewHidden.AllocTensor<DTYPE_NEW_HIDDEN>();
        AscendC::LocalTensor<DTYPE_OUTPUT> outputLocal = outQueueOutput.AllocTensor<DTYPE_OUTPUT>();

        // Concatenate x and hidden
        AscendC::LocalTensor<DTYPE_X> concatLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::Concat(concatLocal, xLocal, hiddenLocal, this->tileLength, this->inputSize, this->hiddenSize);

        // Linear transform: i2h
        AscendC::MatMul(newHiddenLocal, concatLocal, i2hWeightLocal, i2hBiasLocal, this->tileLength, this->inputSize + this->hiddenSize, this->hiddenSize);

        // Tanh activation
        AscendC::Tanh(newHiddenLocal, newHiddenLocal, this->tileLength);

        // Linear transform: h2o
        AscendC::MatMul(outputLocal, newHiddenLocal, h2oWeightLocal, h2oBiasLocal, this->batchSize, this->hiddenSize, this->outputSize);

        outQueueNewHidden.EnQue<DTYPE_NEW_HIDDEN>(newHiddenLocal);
        outQueueOutput.EnQue<DTYPE_OUTPUT>(outputLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueHidden.FreeTensor(hiddenLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_OUTPUT> outputLocal = outQueueOutput.DeQue<DTYPE_OUTPUT>();
        AscendC::LocalTensor<DTYPE_NEW_HIDDEN> newHiddenLocal = outQueueNewHidden.DeQue<DTYPE_NEW_HIDDEN>();
        AscendC::DataCopy(outputGm[progress * this->tileLength], outputLocal, this->tileLength);
        AscendC::DataCopy(newHiddenGm[progress * this->tileLength], newHiddenLocal, this->tileLength);
        outQueueOutput.FreeTensor(outputLocal);
        outQueueNewHidden.FreeTensor(newHiddenLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueHidden, inQueueI2hWeight, inQueueI2hBias, inQueueH2oWeight, inQueueH2oBias;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueOutput, outQueueNewHidden;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_HIDDEN> hiddenGm;
    AscendC::GlobalTensor<DTYPE_I2H_WEIGHT> i2hWeightGm;
    AscendC::GlobalTensor<DTYPE_I2H_BIAS> i2hBiasGm;
    AscendC::GlobalTensor<DTYPE_H2O_WEIGHT> h2oWeightGm;
    AscendC::GlobalTensor<DTYPE_H2O_BIAS> h2oBiasGm;
    AscendC::GlobalTensor<DTYPE_OUTPUT> outputGm;
    AscendC::GlobalTensor<DTYPE_NEW_HIDDEN> newHiddenGm;
    uint32_t batchSize;
    uint32_t inputSize;
    uint32_t hiddenSize;
    uint32_t outputSize;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void vanilla_rnn_custom(GM_ADDR x, GM_ADDR hidden, GM_ADDR i2hWeight, GM_ADDR i2hBias, GM_ADDR h2oWeight, GM_ADDR h2oBias, GM_ADDR output, GM_ADDR newHidden, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelVanillaRnn op;
    op.Init(x, hidden, i2hWeight, i2hBias, h2oWeight, h2oBias, output, newHidden, tiling_data.batchSize, tiling_data.inputSize, tiling_data.hiddenSize, tiling_data.outputSize, tiling_data.tileNum);
    op.Process();
}
