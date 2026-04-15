
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelGemmSigmoidSumLogSumExp {
public:
    __aicore__ inline KernelGemmSigmoidSumLogSumExp() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight1, GM_ADDR bias1,
                                 GM_ADDR weight2, GM_ADDR bias2, GM_ADDR z,
                                 uint32_t batchSize, uint32_t inputSize,
                                 uint32_t hiddenSize, uint32_t outputSize,
                                 uint32_t tileNum)
    {
        this->batchSize = batchSize;
        this->inputSize = inputSize;
        this->hiddenSize = hiddenSize;
        this->outputSize = outputSize;

        uint32_t totalRows = batchSize;
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        
        this->rowsPerBlock = totalRows / blockNum;
        uint32_t remainRows = totalRows % blockNum;
        if (blockIdx < remainRows) {
            this->rowsPerBlock += 1;
            this->startRow = blockIdx * this->rowsPerBlock;
        } else {
            this->startRow = remainRows * (this->rowsPerBlock + 1) + (blockIdx - remainRows) * this->rowsPerBlock;
        }

        xGm.SetGlobalBuffer((__gm__ float *)x, batchSize * inputSize);
        w1Gm.SetGlobalBuffer((__gm__ float *)weight1, hiddenSize * inputSize);
        b1Gm.SetGlobalBuffer((__gm__ float *)bias1, hiddenSize);
        w2Gm.SetGlobalBuffer((__gm__ float *)weight2, outputSize * hiddenSize);
        b2Gm.SetGlobalBuffer((__gm__ float *)bias2, outputSize);
        zGm.SetGlobalBuffer((__gm__ float *)z, batchSize);

        // Allocate buffers for processing
        // We process one row at a time
        // For hidden layer: need inputSize for x_row, hiddenSize for hidden result
        // For output layer: need hiddenSize for sigmoid result, outputSize for output
        uint32_t maxDim = inputSize;
        if (hiddenSize > maxDim) maxDim = hiddenSize;
        if (outputSize > maxDim) maxDim = outputSize;
        
        // Align to 32 bytes (8 floats)
        uint32_t alignedInput = (inputSize + 7) / 8 * 8;
        uint32_t alignedHidden = (hiddenSize + 7) / 8 * 8;
        uint32_t alignedOutput = (outputSize + 7) / 8 * 8;
        
        this->alignedInput = alignedInput;
        this->alignedHidden = alignedHidden;
        this->alignedOutput = alignedOutput;

        pipe.InitBuffer(inQueueX, 1, alignedInput * sizeof(float));
        pipe.InitBuffer(inQueueW, 1, alignedInput * sizeof(float));  // for one row of weight
        pipe.InitBuffer(outQueueH, 1, alignedHidden * sizeof(float));
        pipe.InitBuffer(outQueueO, 1, alignedOutput * sizeof(float));
        pipe.InitBuffer(inQueueB, 1, alignedHidden * sizeof(float));  // reuse for bias
        pipe.InitBuffer(inQueueW2, 1, alignedHidden * sizeof(float)); // for one row of weight2
        pipe.InitBuffer(inQueueB2, 1, alignedOutput * sizeof(float));
    }
    
    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < this->rowsPerBlock; i++) {
            ProcessOneRow(this->startRow + i);
        }
    }

private:
    __aicore__ inline void ProcessOneRow(uint32_t rowIdx)
    {
        // Load x row
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::DataCopy(xLocal, xGm[rowIdx * inputSize], alignedInput);
        inQueueX.EnQue(xLocal);
        xLocal = inQueueX.DeQue<float>();
        
        // Compute hidden = W1 * x + b1 (W1 is [hiddenSize, inputSize])
        AscendC::LocalTensor<float> hLocal = outQueueH.AllocTensor<float>();
        
        // Load bias1
        AscendC::LocalTensor<float> bLocal = inQueueB.AllocTensor<float>();
        AscendC::DataCopy(bLocal, b1Gm[0], alignedHidden);
        inQueueB.EnQue(bLocal);
        bLocal = inQueueB.DeQue<float>();
        AscendC::DataCopy(hLocal, bLocal, alignedHidden);
        inQueueB.FreeTensor(bLocal);
        
        // For each hidden neuron, compute dot product with x
        AscendC::LocalTensor<float> wLocal = inQueueW.AllocTensor<float>();
        for (uint32_t h = 0; h < hiddenSize; h++) {
            AscendC::DataCopy(wLocal, w1Gm[h * inputSize], alignedInput);
            inQueueW.EnQue(wLocal);
            wLocal = inQueueW.DeQue<float>();
            
            // Dot product: sum of element-wise multiply
            // We'll use Mul + ReduceSum
            AscendC::LocalTensor<float> tmpLocal = inQueueB.AllocTensor<float>();
            AscendC::Mul(tmpLocal, xLocal, wLocal, alignedInput);
            
            // Manual reduction
            float dotProduct = 0.0f;
            for (uint32_t k = 0; k < inputSize; k++) {
                dotProduct += tmpLocal.GetValue(k);
            }
            hLocal.SetValue(h, hLocal.GetValue(h) + dotProduct);
            inQueueB.FreeTensor(tmpLocal);
            
            wLocal = inQueueW.AllocTensor<float>();
        }
        inQueueW.FreeTensor(wLocal);
        inQueueX.FreeTensor(xLocal);
        
        // Apply sigmoid to hidden
        AscendC::Sigmoid(hLocal, hLocal, alignedHidden);
        
        // Compute output = W2 * hidden + b2 (W2 is [outputSize, hiddenSize])
        AscendC::LocalTensor<float> oLocal = outQueueO.AllocTensor<float>();
        
        // Load bias2
        AscendC::LocalTensor<float> b2Local = inQueueB2.AllocTensor<float>();
        AscendC::DataCopy(b2Local, b2Gm[0], alignedOutput);
        inQueueB2.EnQue(b2Local);
        b2Local = inQueueB2.DeQue<float>();
        AscendC::DataCopy(oLocal, b2Local, alignedOutput);
        inQueueB2.FreeTensor(b2Local);
        
        AscendC::LocalTensor<float> w2Local = inQueueW2.AllocTensor<float>();
        for (uint32_t o = 0; o < outputSize; o++) {
            AscendC::DataCopy(w2Local, w2Gm[o * hiddenSize], alignedHidden);
            inQueueW2.EnQue(w2Local);
            w2Local = inQueueW2.DeQue<float>();
            
            AscendC::LocalTensor<float> tmpLocal = inQueueB.AllocTensor<float>();
            AscendC::Mul(tmpLocal, hLocal, w2Local, alignedHidden);
            
            float dotProduct = 0.0f;
            for (uint32_t k = 0; k < hiddenSize; k++) {
                dotProduct += tmpLocal.GetValue(k);
            }
            oLocal.SetValue(o, oLocal.GetValue(o) + dotProduct);
            inQueueB.FreeTensor(tmpLocal);
            
            w2Local = inQueueW2.AllocTensor<float>();
        }
        inQueueW2.FreeTensor(w2Local);
        outQueueH.FreeTensor(hLocal);
        
        // Compute LogSumExp over output features
        // logsumexp = log(sum(exp(x_i)))
        // For numerical stability: max + log(sum(exp(x_i - max)))
        float maxVal = oLocal.GetValue(0);
        for (uint32_t o = 1; o < outputSize; o++) {
            float v = oLocal.GetValue(o);
            if (v > maxVal) maxVal = v;
        }
        
        float sumExp = 0.0f;
        for (uint32_t o = 0; o < outputSize; o++) {
            float v = oLocal.GetValue(o);
            float expVal = AscendC::Scalar<float>(v - maxVal);
            // Use exp approximation
            sumExp += expf(v - maxVal);
        }
        
        float lse = maxVal + logf(sumExp);
        
        outQueueO.FreeTensor(oLocal);
        
        // Write result
        // We need to write a single float; use a temporary tensor
        AscendC::LocalTensor<float> resLocal = inQueueB2.AllocTensor<float>();
        resLocal.SetValue(0, lse);
        // Pad remaining to avoid issues
        for (uint32_t k = 1; k < 8; k++) {
            resLocal.SetValue(k, 0.0f);
        }
        AscendC::DataCopy(zGm[rowIdx], resLocal, 8);
        inQueueB2.FreeTensor(resLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX, inQueueW, inQueueB, inQueueW2, inQueueB2;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueH, outQueueO;
    AscendC::GlobalTensor<float> xGm, w1Gm, b1Gm, w2Gm, b2Gm, zGm;
    uint32_t batchSize, inputSize, hiddenSize, outputSize;
    uint32_t rowsPerBlock, startRow;
    uint32_t alignedInput, alignedHidden, alignedOutput;
};

extern "C" __global__ __aicore__ void gemm_sigmoid_sum_log_sum_exp_custom(
    GM_ADDR x, GM_ADDR weight1, GM_ADDR bias1, GM_ADDR weight2, GM_ADDR bias2,
    GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelGemmSigmoidSumLogSumExp op;
    op.Init(x, weight1, bias1, weight2, bias2, z,
            tiling_data.batchSize, tiling_data.inputSize,
            tiling_data.hiddenSize, tiling_data.outputSize,
            tiling_data.tileNum);
    op.Process();
}
