
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelGemmGroupNormHardtanh {
public:
    __aicore__ inline KernelGemmGroupNormHardtanh() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR bias, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling,
                                uint32_t batch_size, uint32_t in_features, uint32_t out_features, uint32_t num_groups,
                                float hardtanh_min, float hardtanh_max)
    {
        this->batch_size = batch_size;
        this->in_features = in_features;
        this->out_features = out_features;
        this->num_groups = num_groups;
        this->hardtanh_min = hardtanh_min;
        this->hardtanh_max = hardtanh_max;
        this->blockLength = out_features / AscendC::GetBlockNum();
        this->tileNum = 4096;
        this->tileLength = this->blockLength / this->tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ float *)x, batch_size * in_features);
        yGm.SetGlobalBuffer((__gm__ float *)y, in_features * out_features);
        biasGm.SetGlobalBuffer((__gm__ float *)bias, out_features);
        zGm.SetGlobalBuffer((__gm__ float *)z, batch_size * out_features);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        // GEMM part
        for (uint32_t i = 0; i < batch_size; i++) {
            for (uint32_t j = 0; j < out_features; j += this->tileLength) {
                CopyIn(i, j);
                ComputeGEMM(i, j);
                CopyOut(i, j);
            }
        }
        // GroupNorm part
        for (uint32_t i = 0; i < batch_size; i++) {
            for (uint32_t j = 0; j < out_features; j += this->tileLength) {
                CopyInNorm(i, j);
                ComputeNorm(i, j);
                CopyOutNorm(i, j);
            }
        }
        // HardTanh part
        for (uint32_t i = 0; i < batch_size; i++) {
            for (uint32_t j = 0; j < out_features; j += this->tileLength) {
                CopyInTanh(i, j);
                ComputeTanh(i, j);
                CopyOutTanh(i, j);
            }
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t batch_idx, uint32_t offset)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::LocalTensor<float> yLocal = inQueueY.AllocTensor<float>();
        AscendC::DataCopy(xLocal, xGm[batch_idx * in_features + offset], this->tileLength);
        AscendC::DataCopy(yLocal, yGm[offset], this->tileLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void ComputeGEMM(uint32_t batch_idx, uint32_t offset)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = inQueueY.DeQue<float>();
        AscendC::LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
        AscendC::MatMul(zLocal, xLocal, yLocal, this->tileLength);
        outQueueZ.EnQue<float>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(uint32_t batch_idx, uint32_t offset)
    {
        AscendC::LocalTensor<float> zLocal = outQueueZ.DeQue<float>();
        AscendC::DataCopy(zGm[batch_idx * out_features + offset], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

    __aicore__ inline void CopyInNorm(uint32_t batch_idx, uint32_t offset)
    {
        AscendC::LocalTensor<float> zLocal = inQueueX.AllocTensor<float>();
        AscendC::DataCopy(zLocal, zGm[batch_idx * out_features + offset], this->tileLength);
        inQueueX.EnQue(zLocal);
    }
    __aicore__ inline void ComputeNorm(uint32_t batch_idx, uint32_t offset)
    {
        AscendC::LocalTensor<float> zLocal = inQueueX.DeQue<float>();
        // Simplified group norm computation
        AscendC::LocalTensor<float> normLocal = outQueueZ.AllocTensor<float>();
        AscendC::GroupNorm(normLocal, zLocal, this->num_groups, this->tileLength);
        outQueueZ.EnQue<float>(normLocal);
        inQueueX.FreeTensor(zLocal);
    }
    __aicore__ inline void CopyOutNorm(uint32_t batch_idx, uint32_t offset)
    {
        AscendC::LocalTensor<float> zLocal = outQueueZ.DeQue<float>();
        AscendC::DataCopy(zGm[batch_idx * out_features + offset], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

    __aicore__ inline void CopyInTanh(uint32_t batch_idx, uint32_t offset)
    {
        AscendC::LocalTensor<float> zLocal = inQueueX.AllocTensor<float>();
        AscendC::DataCopy(zLocal, zGm[batch_idx * out_features + offset], this->tileLength);
        inQueueX.EnQue(zLocal);
    }
    __aicore__ inline void ComputeTanh(uint32_t batch_idx, uint32_t offset)
    {
        AscendC::LocalTensor<float> zLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> tanhLocal = outQueueZ.AllocTensor<float>();
        AscendC::HardTanh(tanhLocal, zLocal, this->hardtanh_min, this->hardtanh_max, this->tileLength);
        outQueueZ.EnQue<float>(tanhLocal);
        inQueueX.FreeTensor(zLocal);
    }
    __aicore__ inline void CopyOutTanh(uint32_t batch_idx, uint32_t offset)
    {
        AscendC::LocalTensor<float> zLocal = outQueueZ.DeQue<float>();
        AscendC::DataCopy(zGm[batch_idx * out_features + offset], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    AscendC::GlobalTensor<float> biasGm;
    AscendC::GlobalTensor<float> zGm;
    uint32_t batch_size;
    uint32_t in_features;
    uint32_t out_features;
    uint32_t num_groups;
    float hardtanh_min;
    float hardtanh_max;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void gemm_group_norm_hardtanh_custom(
    GM_ADDR x, GM_ADDR y, GM_ADDR bias, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelGemmGroupNormHardtanh op;
    op.Init(x, y, bias, z, workspace, tiling,
            tiling_data.batch_size, tiling_data.in_features, tiling_data.out_features,
            tiling_data.num_groups, tiling_data.hardtanh_min, tiling_data.hardtanh_max);
    op.Process();
}
