
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;

class KernelGemmScaleBatchnorm {
public:
    __aicore__ inline KernelGemmScaleBatchnorm() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR scale,
                                 GM_ADDR bn_weight, GM_ADDR bn_bias,
                                 GM_ADDR running_mean, GM_ADDR running_var,
                                 GM_ADDR z, GM_ADDR mean_out, GM_ADDR var_out,
                                 GM_ADDR workspace,
                                 uint32_t batchSize, uint32_t inFeatures, uint32_t outFeatures,
                                 float eps, float momentum, uint32_t isTraining)
    {
        this->batchSize = batchSize;
        this->inFeatures = inFeatures;
        this->outFeatures = outFeatures;
        this->eps = eps;
        this->momentum = momentum;
        this->isTraining = isTraining;

        // Align sizes to 32 bytes (8 floats)
        uint32_t alignedIn = ((inFeatures + 7) / 8) * 8;
        uint32_t alignedOut = ((outFeatures + 7) / 8) * 8;

        xGm.SetGlobalBuffer((__gm__ float*)x, batchSize * inFeatures);
        weightGm.SetGlobalBuffer((__gm__ float*)weight, outFeatures * inFeatures);
        biasGm.SetGlobalBuffer((__gm__ float*)bias, outFeatures);
        scaleGm.SetGlobalBuffer((__gm__ float*)scale, outFeatures);
        bnWeightGm.SetGlobalBuffer((__gm__ float*)bn_weight, outFeatures);
        bnBiasGm.SetGlobalBuffer((__gm__ float*)bn_bias, outFeatures);
        runMeanGm.SetGlobalBuffer((__gm__ float*)running_mean, outFeatures);
        runVarGm.SetGlobalBuffer((__gm__ float*)running_var, outFeatures);
        zGm.SetGlobalBuffer((__gm__ float*)z, batchSize * outFeatures);
        meanOutGm.SetGlobalBuffer((__gm__ float*)mean_out, outFeatures);
        varOutGm.SetGlobalBuffer((__gm__ float*)var_out, outFeatures);

        // We'll use workspace for intermediate gemm+scale output
        gemm_out_Gm.SetGlobalBuffer((__gm__ float*)workspace, batchSize * outFeatures);

        // Allocate buffers
        pipe.InitBuffer(inQueueX, BUFFER_NUM, alignedIn * sizeof(float));
        pipe.InitBuffer(inQueueW, BUFFER_NUM, alignedIn * sizeof(float));
        pipe.InitBuffer(outQueue, BUFFER_NUM, alignedOut * sizeof(float));
        pipe.InitBuffer(tmpBuf, BUFFER_NUM, alignedOut * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        // Step 1: Compute GEMM + scale for each row
        uint32_t alignedIn = ((inFeatures + 7) / 8) * 8;
        uint32_t alignedOut = ((outFeatures + 7) / 8) * 8;

        for (uint32_t b = 0; b < batchSize; b++) {
            // Load bias into output
            AscendC::LocalTensor<float> outLocal = outQueue.AllocTensor<float>();
            // Initialize with bias
            AscendC::DataCopy(outLocal, biasGm[0], alignedOut);
            outQueue.EnQue(outLocal);
            outLocal = outQueue.DeQue<float>();

            // For each output feature, compute dot product
            // We do a simple approach: iterate over input in tiles
            for (uint32_t o = 0; o < outFeatures; o++) {
                float acc = 0.0f;
                // Manual dot product
                AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
                AscendC::LocalTensor<float> wLocal = inQueueW.AllocTensor<float>();

                // Process in chunks
                uint32_t processed = 0;
                while (processed < inFeatures) {
                    uint32_t chunk = inFeatures - processed;
                    if (chunk > alignedIn) chunk = alignedIn;
                    uint32_t alignedChunk = ((chunk + 7) / 8) * 8;

                    AscendC::DataCopy(xLocal, xGm[b * inFeatures + processed], alignedChunk);
                    AscendC::DataCopy(wLocal, weightGm[o * inFeatures + processed], alignedChunk);

                    // Multiply element-wise
                    AscendC::LocalTensor<float> tmpLocal = tmpBuf.AllocTensor<float>();
                    AscendC::Mul(tmpLocal, xLocal, wLocal, alignedChunk);

                    // Sum reduction
                    for (uint32_t i = 0; i < chunk; i++) {
                        acc += tmpLocal.GetValue(i);
                    }
                    tmpBuf.FreeTensor(tmpLocal);
                    processed += chunk;
                }

                inQueueX.FreeTensor(xLocal);
                inQueueW.FreeTensor(wLocal);

                // Add bias
                float val = acc + outLocal.GetValue(o);
                outLocal.SetValue(o, val);
            }

            // Apply scale
            AscendC::LocalTensor<float> scaleLocal = inQueueX.AllocTensor<float>();
            AscendC::DataCopy(scaleLocal, scaleGm[0], alignedOut);
            AscendC::Mul(outLocal, outLocal, scaleLocal, alignedOut);
            inQueueX.FreeTensor(scaleLocal);

            // Store to workspace (gemm_out)
            AscendC::DataCopy(gemm_out_Gm[b * outFeatures], outLocal, alignedOut);
            outQueue.FreeTensor(outLocal);
        }

        // Step 2: BatchNorm
        // Compute mean and variance per feature
        for (uint32_t f = 0; f < outFeatures; f++) {
            float mean = 0.0f;
            for (uint32_t b = 0; b < batchSize; b++) {
                // Read single value
                AscendC::LocalTensor<float> tmpLocal = tmpBuf.AllocTensor<float>();
                uint32_t alignedCopy = 8;
                uint32_t base = b * outFeatures + (f / 8) * 8;
                AscendC::DataCopy(tmpLocal, gemm_out_Gm[base], alignedCopy);
                float val = tmpLocal.GetValue(f % 8);
                tmpBuf.FreeTensor(tmpLocal);
                mean += val;
            }
            mean /= (float)batchSize;

            float var = 0.0f;
            for (uint32_t b = 0; b < batchSize; b++) {
                AscendC::LocalTensor<float> tmpLocal = tmpBuf.AllocTensor<float>();
                uint32_t base = b * outFeatures + (f / 8) * 8;
                AscendC::DataCopy(tmpLocal, gemm_out_Gm[base], 8);
                float val = tmpLocal.GetValue(f % 8);
                tmpBuf.FreeTensor(tmpLocal);
                float diff = val - mean;
                var += diff * diff;
            }
            var /= (float)batchSize;

            // Store mean and var
            AscendC::LocalTensor<float> meanLocal = tmpBuf.AllocTensor<float>();
            meanLocal.SetValue(0, mean);
            // We'll write them one at a time using a trick
            tmpBuf.FreeTensor(meanLocal);

            // Read bn_weight and bn_bias for this feature
            AscendC::LocalTensor<float> bnParamLocal = inQueueX.AllocTensor<float>();
            uint32_t bnBase = (f / 8) * 8;
            AscendC::DataCopy(bnParamLocal, bnWeightGm[bnBase], 8);
            float gamma = bnParamLocal.GetValue(f % 8);
            AscendC::DataCopy(bnParamLocal, bnBiasGm[bnBase], 8);
            float beta = bnParamLocal.GetValue(f % 8);
            inQueueX.FreeTensor(bnParamLocal);

            float invstd = 1.0f / sqrtf(var + eps);

            // Apply BN: y = gamma * (x - mean) * invstd + beta
            for (uint32_t b = 0; b < batchSize; b++) {
                AscendC::LocalTensor<float> tmpLocal = tmpBuf.AllocTensor<float>();
                uint32_t base = b * outFeatures + (f / 8) * 8;
                AscendC::DataCopy(tmpLocal, gemm_out_Gm[base], 8);
                float val = tmpLocal.GetValue(f % 8);
                float normed = gamma * (val - mean) * invstd + beta;

                // Write to output
                AscendC::LocalTensor<float> outLocal2 = outQueue.AllocTensor<float>();
                uint32_t outBase = b * outFeatures + (f / 8) * 8;
                AscendC::DataCopy(outLocal2, zGm[outBase], 8);
                outLocal2.SetValue(f % 8, normed);
                AscendC::DataCopy(zGm[outBase], outLocal2, 8);
                outQueue.FreeTensor(outLocal2);

                tmpBuf.FreeTensor(tmpLocal);
            }

            // Update running stats
            if (isTraining) {
                AscendC::LocalTensor<float> runLocal = inQueueX.AllocTensor<float>();
                AscendC::DataCopy(runLocal, runMeanGm[bnBase], 8);
                float rm = runLocal.GetValue(f % 8);
                rm = (1.0f - momentum) * rm + momentum * mean;
                runLocal.SetValue(f % 8, rm);
                AscendC::DataCopy(runMeanGm[bnBase], runLocal, 8);

                AscendC::DataCopy(runLocal, runVarGm[bnBase], 8);
                float rv = runLocal.GetValue(f % 8);
                float unbiased_var = var * (float)batchSize / (float)(batchSize - 1);
                rv = (1.0f - momentum) * rv + momentum * unbiased_var;
                runLocal.SetValue(f % 8, rv);
                AscendC::DataCopy(runVarGm[bnBase], runLocal, 8);
                inQueueX.FreeTensor(runLocal);
            }

            // Store mean_out, var_out
            AscendC::LocalTensor<float> mvLocal = inQueueX.AllocTensor<float>();
            AscendC::DataCopy(mvLocal, meanOutGm[bnBase], 8);
            mvLocal.SetValue(f % 8, mean);
            AscendC::DataCopy(meanOutGm[bnBase], mvLocal, 8);
            AscendC::DataCopy(mvLocal, varOutGm[bnBase], 8);
            mvLocal.SetValue(f % 8, var);
            AscendC::DataCopy(varOutGm[bnBase], mvLocal, 8);
            inQueueX.FreeTensor(mvLocal);
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueW;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueue;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> tmpBuf;
    AscendC::GlobalTensor<float> xGm, weightGm, biasGm, scaleGm;
    AscendC::GlobalTensor<float> bnWeightGm, bnBiasGm, runMeanGm, runVarGm;
    AscendC::GlobalTensor<float> zGm, meanOutGm, varOutGm;
    AscendC::GlobalTensor<float> gemm_out_Gm;
    uint32_t batchSize, inFeatures, outFeatures;
    float eps, momentum;
    uint32_t isTraining;

    __aicore__ inline float sqrtf(float x) {
        // Newton's method
        if (x <= 0.0f) return 0.0f;
        float r = x;
        for (int i = 0; i < 10; i++) {
            r = 0.5f * (r + x / r);
        }
        return r;
    }
};

extern "C" __global__ __aicore__ void gemm_scale_batchnorm_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR scale,
    GM_ADDR bn_weight, GM_ADDR bn_bias, GM_ADDR running_mean, GM_ADDR running_var,
    GM_ADDR z, GM_ADDR mean_out, GM_ADDR var_out,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelGemmScaleBatchnorm op;
    op.Init(x, weight, bias, scale, bn_weight, bn_bias, running_mean, running_var,
            z, mean_out, var_out, workspace,
            tiling_data.batchSize, tiling_data.inFeatures, tiling_data.outFeatures,
            tiling_data.eps, tiling_data.momentum, tiling_data.isTraining);
    op.Process();
}
