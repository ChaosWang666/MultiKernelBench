
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelMatVecMul {
public:
    __aicore__ inline KernelMatVecMul() {}
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, uint32_t M, uint32_t K, uint32_t tileK)
    {
        this->M = M;
        this->K = K;
        this->tileK = tileK;
        
        // Each block handles a subset of rows
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        this->rowsPerBlock = M / blockNum;
        this->startRow = blockIdx * this->rowsPerBlock;
        
        // Handle remainder rows in last block
        if (blockIdx == blockNum - 1) {
            this->rowsPerBlock = M - this->startRow;
        }
        
        aGm.SetGlobalBuffer((__gm__ float *)a + this->startRow * K, this->rowsPerBlock * K);
        bGm.SetGlobalBuffer((__gm__ float *)b, K);
        cGm.SetGlobalBuffer((__gm__ float *)c + this->startRow, this->rowsPerBlock);
        
        // Allocate buffers for tileK elements
        pipe.InitBuffer(inQueueA, BUFFER_NUM, tileK * sizeof(float));
        pipe.InitBuffer(inQueueB, BUFFER_NUM, tileK * sizeof(float));
        pipe.InitBuffer(outQueueC, 1, tileK * sizeof(float));
    }
    
    __aicore__ inline void Process()
    {
        uint32_t kTiles = K / tileK;
        
        for (uint32_t row = 0; row < this->rowsPerBlock; row++) {
            float rowSum = 0.0f;
            
            for (uint32_t kt = 0; kt < kTiles; kt++) {
                // Copy in a tile of the row
                AscendC::LocalTensor<float> aLocal = inQueueA.AllocTensor<float>();
                AscendC::DataCopy(aLocal, aGm[row * K + kt * tileK], tileK);
                inQueueA.EnQue(aLocal);
                
                // Copy in corresponding tile of vector b
                AscendC::LocalTensor<float> bLocal = inQueueB.AllocTensor<float>();
                AscendC::DataCopy(bLocal, bGm[kt * tileK], tileK);
                inQueueB.EnQue(bLocal);
                
                // Compute element-wise multiply
                AscendC::LocalTensor<float> aComp = inQueueA.DeQue<float>();
                AscendC::LocalTensor<float> bComp = inQueueB.DeQue<float>();
                AscendC::LocalTensor<float> cLocal = outQueueC.AllocTensor<float>();
                
                AscendC::Mul(cLocal, aComp, bComp, tileK);
                
                // Reduce sum
                float tileSum = 0.0f;
                // Use ReduceSum to sum up all elements
                AscendC::LocalTensor<float> workLocal = cLocal;
                
                // Manual reduction in stages
                uint32_t len = tileK;
                while (len > 64) {
                    uint32_t half = len / 2;
                    AscendC::Add(workLocal, workLocal, workLocal[half], half);
                    len = half;
                }
                // Final reduction for remaining elements
                for (uint32_t i = 1; i < len; i++) {
                    workLocal.SetValue(0, workLocal.GetValue(0) + workLocal.GetValue(i));
                }
                tileSum = workLocal.GetValue(0);
                
                rowSum += tileSum;
                
                outQueueC.FreeTensor(cLocal);
                inQueueA.FreeTensor(aComp);
                inQueueB.FreeTensor(bComp);
            }
            
            // Write result - need to use DataCopy with at least 32 bytes (8 floats)
            // Use a temporary tensor approach
            AscendC::LocalTensor<float> tmpOut = inQueueA.AllocTensor<float>();
            // Zero out first 8 elements
            for (uint32_t i = 0; i < 8; i++) {
                tmpOut.SetValue(i, 0.0f);
            }
            tmpOut.SetValue(0, rowSum);
            
            // We need to write just one float - use SetValue on global memory
            // DataCopy requires alignment, so we copy 8 floats but only first matters
            // This is safe as long as we allocated enough output space
            // Actually, we should be more careful. Let's accumulate all rows first.
            // For simplicity we do single float writes if possible
            cGm.SetValue(row, rowSum);
            
            inQueueA.FreeTensor(tmpOut);
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueA, inQueueB;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueC;
    AscendC::GlobalTensor<float> aGm;
    AscendC::GlobalTensor<float> bGm;
    AscendC::GlobalTensor<float> cGm;
    uint32_t M;
    uint32_t K;
    uint32_t tileK;
    uint32_t rowsPerBlock;
    uint32_t startRow;
};

extern "C" __global__ __aicore__ void matrix_vector_multiplication_custom(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelMatVecMul op;
    op.Init(a, b, c, tiling_data.M, tiling_data.K, tiling_data.tileK);
    op.Process();
}
