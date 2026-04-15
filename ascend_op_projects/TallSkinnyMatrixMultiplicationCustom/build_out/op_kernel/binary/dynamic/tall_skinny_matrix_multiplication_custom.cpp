
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;

class KernelMatmul {
public:
    __aicore__ inline KernelMatmul() {}
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, uint32_t M, uint32_t N, uint32_t K)
    {
        this->M = M;
        this->N = N;
        this->K = K;
        
        // Each block handles a chunk of rows of the output C (M x N)
        // A is (M x K), B is (K x N), C is (M x N)
        // For tall-skinny: M is large, K and N are small
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        
        // Divide M rows among blocks
        this->rowsPerBlock = (M + blockNum - 1) / blockNum;
        this->startRow = blockIdx * this->rowsPerBlock;
        this->endRow = startRow + rowsPerBlock;
        if (this->endRow > M) {
            this->endRow = M;
        }
        this->actualRows = (this->endRow > this->startRow) ? (this->endRow - this->startRow) : 0;
        
        aGm.SetGlobalBuffer((__gm__ float *)a, M * K);
        bGm.SetGlobalBuffer((__gm__ float *)b, K * N);
        cGm.SetGlobalBuffer((__gm__ float *)c, M * N);
        
        // We process rows in tiles. Each tile processes some rows at a time.
        // K and N are small (32 each in the given example), so we can load entire rows.
        // We'll process multiple rows at a time to improve efficiency.
        // tileRows: how many rows to process per tile iteration
        // We need: tileRows * K floats for A tile, K * N floats for B, tileRows * N floats for C tile
        // Plus temporary buffers for computation
        // UB size is limited, let's be conservative
        // K=32, N=32768 or K=32, N=32768... actually M=32768, N=32, K=32
        // So A row = K=32 floats, C row = N=32768 floats
        // Wait, let me re-read: M=32768, K=32 (since A is MxK=32768x32, B is KxN=32x32768)
        // Actually from the problem: M=32768, N=32, A=(M,N)=(32768,32), B=(N,M)=(32,32768)
        // So matmul(A,B) = (32768,32) x (32,32768) = (32768,32768)
        // Wait no. Let me re-read. M=16384*2=32768, N=16*2=32
        // A = torch.rand(M, N) = (32768, 32)
        // B = torch.rand(N, M) = (32, 32768)
        // C = A @ B = (32768, 32768)
        // So K=32 (inner dim), output is 32768 x 32768
        
        // For this kernel: A is (M, K), B is (K, N_out) where K is small
        // M = 32768, K = 32, N_out = 32768
        // Each block handles actualRows rows of output. Each output row has N elements.
        // For each row i of output: C[i,:] = sum over k: A[i,k] * B[k,:]
        // B[k,:] has N elements.
        
        // We'll process one output row at a time, or a few columns at a time
        // N can be large (32768), K is small (32)
        // For each row, we need to load A[i, 0..K-1] (K floats) and compute dot with B columns
        // Actually we can do: for each row i, for a chunk of columns j..j+tileN:
        //   C[i, j..j+tileN] = sum_k A[i,k] * B[k, j..j+tileN]
        // We can vectorize over the column chunk.

        // Let's pick a tile size for columns
        // We need in UB: K * tileN floats for B chunk, tileN floats for C chunk, K floats for A row, tileN floats temp
        // With UB ~ 192KB = 49152 floats
        // Let tileN = 1024: K*tileN = 32*1024 = 32768, + 1024 + 32 + 1024 ~ 34848 floats ~ 136KB. OK.
        
        this->tileN = 1024;
        if (this->tileN > N) this->tileN = N;
        // Align tileN to 8 (32 bytes / 4 bytes per float)
        this->tileN = (this->tileN / 8) * 8;
        if (this->tileN == 0) this->tileN = 8;
        
        uint32_t bBufSize = K * this->tileN * sizeof(float);
        uint32_t aBufSize = K * sizeof(float);
        // Align to 32 bytes
        aBufSize = ((aBufSize + 31) / 32) * 32;
        uint32_t cBufSize = this->tileN * sizeof(float);
        uint32_t tempBufSize = this->tileN * sizeof(float);
        
        pipe.InitBuffer(bQueue, 1, bBufSize);
        pipe.InitBuffer(aQueue, 1, aBufSize);
        pipe.InitBuffer(cQueue, 1, cBufSize);
        pipe.InitBuffer(tempBuf, tempBufSize);
    }
    
    __aicore__ inline void Process()
    {
        if (this->actualRows == 0) return;
        
        uint32_t colTiles = (N + tileN - 1) / tileN;
        
        for (uint32_t ct = 0; ct < colTiles; ct++) {
            uint32_t colStart = ct * tileN;
            uint32_t curTileN = tileN;
            if (colStart + curTileN > N) curTileN = N - colStart;
            
            // Load B chunk: B[0..K-1, colStart..colStart+curTileN-1]
            // B is stored row-major: B[k, j] at offset k*N + j
            AscendC::LocalTensor<float> bLocal = bQueue.AllocTensor<float>();
            for (uint32_t k = 0; k < K; k++) {
                AscendC::DataCopy(bLocal[k * curTileN], bGm[k * N + colStart], curTileN);
            }
            bQueue.EnQue(bLocal);
            bLocal = bQueue.DeQue<float>();
            
            for (uint32_t r = 0; r < actualRows; r++) {
                uint32_t globalRow = startRow + r;
                
                // Load A[globalRow, 0..K-1]
                AscendC::LocalTensor<float> aLocal = aQueue.AllocTensor<float>();
                uint32_t aElems = ((K + 7) / 8) * 8;
                AscendC::DataCopy(aLocal, aGm[globalRow * K], aElems);
                aQueue.EnQue(aLocal);
                aLocal = aQueue.DeQue<float>();
                
                // Compute C[globalRow, colStart..colStart+curTileN-1]
                AscendC::LocalTensor<float> cLocal = cQueue.AllocTensor<float>();
                AscendC::LocalTensor<float> tmpLocal = tempBuf.Get<float>();
                
                // Initialize cLocal to 0
                AscendC::Duplicate(cLocal, (float)0.0f, curTileN);
                
                for (uint32_t k = 0; k < K; k++) {
                    float aVal = aLocal.GetValue(k);
                    // tmpLocal = aVal * B[k, colStart..colStart+curTileN]
                    AscendC::Muls(tmpLocal, bLocal[k * curTileN], aVal, curTileN);
                    AscendC::Add(cLocal, cLocal, tmpLocal, curTileN);
                }
                
                // Store result
                cQueue.EnQue(cLocal);
                cLocal = cQueue.DeQue<float>();
                AscendC::DataCopy(cGm[globalRow * N + colStart], cLocal, curTileN);
                
                aQueue.FreeTensor(aLocal);
                cQueue.FreeTensor(cLocal);
            }
            
            bQueue.FreeTensor(bLocal);
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> bQueue, aQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> cQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tempBuf;
    AscendC::GlobalTensor<float> aGm;
    AscendC::GlobalTensor<float> bGm;
    AscendC::GlobalTensor<float> cGm;
    uint32_t M, N, K;
    uint32_t rowsPerBlock, startRow, endRow, actualRows;
    uint32_t tileN;
};

extern "C" __global__ __aicore__ void tall_skinny_matrix_multiplication_custom(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelMatmul op;
    op.Init(a, b, c, tiling_data.M, tiling_data.N, tiling_data.K);
    op.Process();
}
