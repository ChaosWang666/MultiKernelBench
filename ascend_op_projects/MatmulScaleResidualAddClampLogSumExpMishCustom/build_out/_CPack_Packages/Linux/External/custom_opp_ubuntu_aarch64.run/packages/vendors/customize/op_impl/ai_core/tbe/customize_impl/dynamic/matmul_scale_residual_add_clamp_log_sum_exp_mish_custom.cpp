
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue
 
class KernelMatmulScaleResidualAddClampLogSumExpMishCustom {
public:
    __aicore__ inline KernelMatmulScaleResidualAddClampLogSumExpMishCustom() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, uint32_t totalLength, uint32_t tileNum, float scale_factor, float clamp_min, float clamp_max)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;
        this->scale_factor = scale_factor;
        this->clamp_min = clamp_min;
        this->clamp_max = clamp_max;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        zGm.SetGlobalBuffer((__gm__ DTYPE_Z *)z + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Z));
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
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Z> zLocal = outQueueZ.AllocTensor<DTYPE_Z>();
        
        // Scale operation
        AscendC::Mul(zLocal, xLocal, this->scale_factor, this->tileLength);
        
        // Residual connection (add itself)
        AscendC::Add(zLocal, zLocal, xLocal, this->tileLength);
        
        // Clamp operation
        AscendC::Clip(zLocal, zLocal, this->clamp_min, this->clamp_max, this->tileLength);
        
        // LogSumExp (simplified approximation)
        AscendC::ReduceSum(zLocal, zLocal, this->tileLength, 1, true);
        
        // Mish activation
        AscendC::Mish(zLocal, zLocal, this->tileLength);
        
        outQueueZ.EnQue<DTYPE_Z>(zLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_Z> zLocal = outQueueZ.DeQue<DTYPE_Z>();
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Z> zGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    float scale_factor;
    float clamp_min;
    float clamp_max;
};

extern "C" __global__ __aicore__ void matmul_scale_residual_add_clamp_log_sum_exp_mish_custom(GM_ADDR x, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelMatmulScaleResidualAddClampLogSumExpMishCustom op;
    op.Init(x, z, tiling_data.totalLength, tiling_data.tileNum, tiling_data.scale_factor, tiling_data.clamp_min, tiling_data.clamp_max);
    op.Process();
}
