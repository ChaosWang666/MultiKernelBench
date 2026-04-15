
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;

class KernelMatmul {
public:
    __aicore__ inline KernelMatmul() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, 
                                 uint32_t M, uint32_t N, uint32_t K, uint32_t tileK)
    {
        this->M = M;
        this->N = N;
        this->K = K;
        this->tileK = tileK;
        
        // Each block handles a subset of (M, N) output elements
        // We distribute output rows across blocks
        uint32_t totalBlocks = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        
        // Total output elements = M * N
        // Each block handles some rows of M
        this->rowsPerBlock = (M + totalBlocks - 1) / totalBlocks;
        this->startRow = blockIdx * this->rowsPerBlock;
        if (this->startRow + this->rowsPerBlock > M) {
            this->rowsPerBlock = (this->startRow < M) ? (M - this->startRow) : 0;
        }
        
        xGm.SetGlobalBuffer((__gm__ float *)x, M * K);
        yGm.SetGlobalBuffer((__gm__ float *)y, K * N);
        zGm.SetGlobalBuffer((__gm__ float *)z, M * N);
        
        // We process one row of output at a time, computing dot products
        // For each output element C[i][j] = sum_k A[i][k] * B[k][j]
        // We tile along K with tileK chunk size
        // We load a tile of A row (tileK elements) and corresponding tileK rows of B (tileK x N -> we pick column j)
        
        // Buffer for a tile of A: tileK floats
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileK * sizeof(float));
        // Buffer for a tile of B column: tileK floats  
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileK * sizeof(float));
        // Buffer for partial result and temp
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileK * sizeof(float));
    }
    
    __aicore__ inline void Process()
    {
        if (this->rowsPerBlock == 0) return;
        
        for (uint32_t i = 0; i < this->rowsPerBlock; i++) {
            uint32_t row = this->startRow + i;
            for (uint32_t j = 0; j < N; j++) {
                float acc = 0.0f;
                uint32_t kTiles = (K + tileK - 1) / tileK;
                for (uint32_t kt = 0; kt < kTiles; kt++) {
                    uint32_t kStart = kt * tileK;
                    uint32_t kLen = tileK;
                    if (kStart + kLen > K) kLen = K - kStart;
                    
                    // Align kLen up to 8 for AscendC requirements
                    uint32_t kLenAligned = (kLen + 7) / 8 * 8;
                    
                    AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
                    AscendC::LocalTensor<float> yLocal = inQueueY.AllocTensor<float>();
                    AscendC::LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
                    
                    // Copy A[row][kStart:kStart+kLen]
                    // A is row-major: A[row][k] = xGm[row * K + k]
                    AscendC::DataCopy(xLocal, xGm[row * K + kStart], kLenAligned);
                    
                    // Copy B[kStart:kStart+kLen][j] - this is strided access
                    // B is row-major: B[k][j] = yGm[k * N + j]
                    // We need to gather kLen elements with stride N
                    // Use DataCopyParams for strided copy
                    // DataCopy with DataCopyParams: nBurstLen, srcStride, dstStride, nBurst
                    // Each burst copies 1 block (32 bytes = 8 floats)
                    // For single element copy per burst, we need special handling
                    
                    // Since strided gather from B column is not trivially supported 
                    // with simple DataCopy, we'll copy one element at a time
                    // Actually for better perf let's handle it carefully
                    
                    // For column access, copy element by element to local buffer
                    for (uint32_t kk = 0; kk < kLen; kk++) {
                        yLocal.SetValue(kk, yGm.GetValue(((kStart + kk) * N) + j));
                    }
                    // Zero pad
                    for (uint32_t kk = kLen; kk < kLenAligned; kk++) {
                        yLocal.SetValue(kk, 0.0f);
                    }
                    
                    // Multiply element-wise
                    AscendC::Mul(zLocal, xLocal, yLocal, kLenAligned);
                    
                    // Sum reduction
                    // We need to sum all elements in zLocal
                    float partialSum = 0.0f;
                    for (uint32_t kk = 0; kk < kLen; kk++) {
                        partialSum += zLocal.GetValue(kk);
                    }
                    acc += partialSum;
                    
                    outQueueZ.FreeTensor(zLocal);
                    inQueueX.FreeTensor(xLocal);
                    inQueueY.FreeTensor(yLocal);
                }
                // Write result
                zGm.SetValue(row * N + j, acc);
            }
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    AscendC::GlobalTensor<float> zGm;
    uint32_t M, N, K;
    uint32_t tileK;
    uint32_t rowsPerBlock;
    uint32_t startRow;
};

extern "C" __global__ __aicore__ void matmul_with_large_k_dimension_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelMatmul op;
    op.Init(x, y, z, tiling_data.M, tiling_data.N, tiling_data.K, tiling_data.tileK);
    op.Process();
}
