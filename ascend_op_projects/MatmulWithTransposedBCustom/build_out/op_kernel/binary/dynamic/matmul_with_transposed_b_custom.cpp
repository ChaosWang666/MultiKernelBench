
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;
// Tile size for K dimension processing
constexpr int32_t K_TILE = 256;

class KernelMatmulTransB {
public:
    __aicore__ inline KernelMatmulTransB() {}
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, uint32_t M, uint32_t N, uint32_t K)
    {
        this->M = M;
        this->N = N;
        this->K = K;
        
        // Total work items: M * N output elements
        // Each block processes a chunk of rows of C
        uint32_t totalRows = M;
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        
        this->rowsPerBlock = (totalRows + blockNum - 1) / blockNum;
        this->startRow = blockIdx * this->rowsPerBlock;
        this->endRow = (this->startRow + this->rowsPerBlock > M) ? M : (this->startRow + this->rowsPerBlock);
        if (this->startRow >= M) {
            this->startRow = M;
            this->endRow = M;
        }
        this->actualRows = this->endRow - this->startRow;
        
        aGm.SetGlobalBuffer((__gm__ float *)a, M * K);
        bGm.SetGlobalBuffer((__gm__ float *)b, N * K);
        cGm.SetGlobalBuffer((__gm__ float *)c, M * N);
        
        // Allocate buffers for tiles of A, B and partial results
        // We process one row of A at a time, and tile over K and N
        uint32_t kTile = (K < K_TILE) ? K : K_TILE;
        // Align to 32 bytes = 8 floats
        uint32_t kTileAligned = ((kTile + 7) / 8) * 8;
        
        pipe.InitBuffer(inQueueA, BUFFER_NUM, kTileAligned * sizeof(float));
        pipe.InitBuffer(inQueueB, BUFFER_NUM, kTileAligned * sizeof(float));
        pipe.InitBuffer(outQueueC, BUFFER_NUM, 8 * sizeof(float)); // single element output, aligned
    }
    
    __aicore__ inline void Process()
    {
        if (this->actualRows == 0) return;
        
        uint32_t kTile = (K < K_TILE) ? K : K_TILE;
        uint32_t kTileAligned = ((kTile + 7) / 8) * 8;
        
        for (uint32_t row = this->startRow; row < this->endRow; row++) {
            for (uint32_t col = 0; col < N; col++) {
                // Compute C[row][col] = sum_k A[row][k] * B[col][k]
                // B is (N, K), so B[col][k] = bGm[col * K + k]
                float accumulator = 0.0f;
                
                for (uint32_t kStart = 0; kStart < K; kStart += kTile) {
                    uint32_t kLen = ((kStart + kTile) > K) ? (K - kStart) : kTile;
                    uint32_t kLenAligned = ((kLen + 7) / 8) * 8;
                    
                    // Copy A tile
                    AscendC::LocalTensor<float> aLocal = inQueueA.AllocTensor<float>();
                    // Zero out the buffer first if kLen < kLenAligned
                    if (kLen < kLenAligned) {
                        AscendC::Duplicate<float>(aLocal, 0.0f, kLenAligned);
                    }
                    AscendC::DataCopy(aLocal, aGm[row * K + kStart], kLenAligned);
                    inQueueA.EnQue(aLocal);
                    
                    // Copy B tile 
                    AscendC::LocalTensor<float> bLocal = inQueueB.AllocTensor<float>();
                    if (kLen < kLenAligned) {
                        AscendC::Duplicate<float>(bLocal, 0.0f, kLenAligned);
                    }
                    AscendC::DataCopy(bLocal, bGm[col * K + kStart], kLenAligned);
                    inQueueB.EnQue(bLocal);
                    
                    // Compute dot product
                    AscendC::LocalTensor<float> aComp = inQueueA.DeQue<float>();
                    AscendC::LocalTensor<float> bComp = inQueueB.DeQue<float>();
                    AscendC::LocalTensor<float> cLocal = outQueueC.AllocTensor<float>();
                    
                    // Element-wise multiply
                    AscendC::Mul(aComp, aComp, bComp, kLenAligned);
                    
                    // Reduce sum - use ReduceSum
                    float partialSum = 0.0f;
                    AscendC::ReduceSum(cLocal, aComp, cLocal, kLenAligned);
                    partialSum = cLocal.GetValue(0);
                    
                    accumulator += partialSum;
                    
                    outQueueC.FreeTensor(cLocal);
                    inQueueA.FreeTensor(aComp);
                    inQueueB.FreeTensor(bComp);
                }
                
                // Write result
                AscendC::LocalTensor<float> cOut = outQueueC.AllocTensor<float>();
                cOut.SetValue(0, accumulator);
                // Pad remaining values
                for (int i = 1; i < 8; i++) {
                    cOut.SetValue(i, 0.0f);
                }
                outQueueC.EnQue(cOut);
                AscendC::LocalTensor<float> cWrite = outQueueC.DeQue<float>();
                AscendC::DataCopy(cGm[row * N + col], cWrite, 8);
                outQueueC.FreeTensor(cWrite);
            }
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueA, inQueueB;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueC;
    AscendC::GlobalTensor<float> aGm;
    AscendC::GlobalTensor<float> bGm;
    AscendC::GlobalTensor<float> cGm;
    uint32_t M, N, K;
    uint32_t rowsPerBlock;
    uint32_t startRow, endRow, actualRows;
};

extern "C" __global__ __aicore__ void matmul_with_transposed_b_custom(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelMatmulTransB op;
    op.Init(a, b, c, tiling_data.M, tiling_data.N, tiling_data.K);
    op.Process();
}
