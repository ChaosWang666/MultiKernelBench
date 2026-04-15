
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;

class KernelMatmul {
public:
    __aicore__ inline KernelMatmul() {}
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, 
                                 uint32_t M, uint32_t N, uint32_t K,
                                 uint32_t tileM, uint32_t tileN)
    {
        this->M = M;
        this->N = N;
        this->K = K;
        this->tileM = tileM;
        this->tileN = tileN;
        
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();
        
        // Each block handles a contiguous chunk of rows
        this->startRow = blockIdx * tileM;
        this->endRow = (blockIdx + 1) * tileM;
        if (this->endRow > M) {
            this->endRow = M;
        }
        this->myRows = (this->endRow > this->startRow) ? (this->endRow - this->startRow) : 0;
        
        aGm.SetGlobalBuffer((__gm__ float *)a, M * K);
        bGm.SetGlobalBuffer((__gm__ float *)b, K * N);
        cGm.SetGlobalBuffer((__gm__ float *)c, M * N);
        
        // We need buffers for:
        // A tile: myRows x K (but we process row by row or small chunks)
        // B tile: K x tileN
        // C tile: tileN (accumulator for one row)
        // Process one row at a time, tile over N
        // For each row: load A row (K floats), then for each N tile load B columns and compute
        
        // Align tileN to 8 (32 bytes / 4 bytes per float)
        if (this->tileN % 8 != 0) {
            this->tileN = ((this->tileN + 7) / 8) * 8;
        }
        
        // Buffer sizes
        uint32_t aRowSize = ((K + 7) / 8) * 8; // aligned K
        this->alignedK = aRowSize;
        
        pipe.InitBuffer(inQueueA, BUFFER_NUM, aRowSize * sizeof(float));
        pipe.InitBuffer(inQueueB, BUFFER_NUM, this->tileN * sizeof(float));
        pipe.InitBuffer(outQueueC, BUFFER_NUM, this->tileN * sizeof(float));
        pipe.InitBuffer(tmpBuf, BUFFER_NUM, this->tileN * sizeof(float));
    }
    
    __aicore__ inline void Process()
    {
        if (this->myRows == 0) return;
        
        for (uint32_t row = 0; row < this->myRows; row++) {
            uint32_t globalRow = this->startRow + row;
            
            for (uint32_t nStart = 0; nStart < N; nStart += this->tileN) {
                uint32_t curTileN = this->tileN;
                if (nStart + curTileN > N) {
                    curTileN = N - nStart;
                }
                uint32_t alignedTileN = ((curTileN + 7) / 8) * 8;
                
                // Allocate output buffer and zero it
                AscendC::LocalTensor<float> cLocal = outQueueC.AllocTensor<float>();
                AscendC::Duplicate(cLocal, (float)0.0f, alignedTileN);
                
                // For each k, load A[row][k] and B[k][nStart:nStart+curTileN]
                for (uint32_t k = 0; k < K; k++) {
                    // Load B[k][nStart .. nStart+curTileN-1]
                    AscendC::LocalTensor<float> bLocal = inQueueB.AllocTensor<float>();
                    AscendC::DataCopy(bLocal, bGm[k * N + nStart], alignedTileN);
                    inQueueB.EnQue(bLocal);
                    bLocal = inQueueB.DeQue<float>();
                    
                    // Load scalar A[globalRow][k]
                    float aVal;
                    // We need to load A value - load a small aligned block
                    AscendC::LocalTensor<float> aLocal = inQueueA.AllocTensor<float>();
                    uint32_t aAlignedOffset = (k / 8) * 8;
                    AscendC::DataCopy(aLocal, aGm[globalRow * K + aAlignedOffset], this->alignedK <= 8 ? 8 : this->alignedK);
                    inQueueA.EnQue(aLocal);
                    aLocal = inQueueA.DeQue<float>();
                    aVal = aLocal.GetValue(k - aAlignedOffset);
                    inQueueA.FreeTensor(aLocal);
                    
                    // cLocal += aVal * bLocal
                    AscendC::LocalTensor<float> tLocal = tmpBuf.AllocTensor<float>();
                    AscendC::Muls(tLocal, bLocal, aVal, alignedTileN);
                    AscendC::Add(cLocal, cLocal, tLocal, alignedTileN);
                    tmpBuf.FreeTensor(tLocal);
                    
                    inQueueB.FreeTensor(bLocal);
                }
                
                // Write out result
                outQueueC.EnQue(cLocal);
                cLocal = outQueueC.DeQue<float>();
                AscendC::DataCopy(cGm[globalRow * N + nStart], cLocal, alignedTileN);
                outQueueC.FreeTensor(cLocal);
            }
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueA, inQueueB;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueC;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    AscendC::GlobalTensor<float> aGm;
    AscendC::GlobalTensor<float> bGm;
    AscendC::GlobalTensor<float> cGm;
    uint32_t M, N, K;
    uint32_t tileM, tileN;
    uint32_t startRow, endRow, myRows;
    uint32_t alignedK;
};

extern "C" __global__ __aicore__ void matmul_with_small_k_dimension_custom(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelMatmul op;
    op.Init(a, b, c, tiling_data.M, tiling_data.N, tiling_data.K, tiling_data.tileM, tiling_data.tileN);
    op.Process();
}
