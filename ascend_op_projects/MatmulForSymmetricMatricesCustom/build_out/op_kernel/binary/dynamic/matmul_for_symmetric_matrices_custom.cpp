
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;

class KernelMatmulSymmetric {
public:
    __aicore__ inline KernelMatmulSymmetric() {}
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, uint32_t N, uint32_t tileM, uint32_t tileN, uint32_t tileK)
    {
        this->N = N;
        this->tileM = tileM;
        this->tileN = tileN;
        this->tileK = tileK;

        // Total number of output tiles
        uint32_t tilesInM = (N + tileM - 1) / tileM;
        uint32_t tilesInN = (N + tileN - 1) / tileN;
        uint32_t totalTiles = tilesInM * tilesInN;
        
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        
        // Each block handles a subset of output tiles
        this->tilesPerBlock = (totalTiles + blockNum - 1) / blockNum;
        this->startTile = blockIdx * this->tilesPerBlock;
        this->endTile = startTile + this->tilesPerBlock;
        if (this->endTile > totalTiles) {
            this->endTile = totalTiles;
        }
        this->tilesInM = tilesInM;
        this->tilesInN = tilesInN;

        aGm.SetGlobalBuffer((__gm__ float *)a, N * N);
        bGm.SetGlobalBuffer((__gm__ float *)b, N * N);
        cGm.SetGlobalBuffer((__gm__ float *)c, N * N);

        // Allocate local buffers for tiles
        pipe.InitBuffer(inQueueA, BUFFER_NUM, tileM * tileK * sizeof(float));
        pipe.InitBuffer(inQueueB, BUFFER_NUM, tileK * tileN * sizeof(float));
        pipe.InitBuffer(outQueueC, BUFFER_NUM, tileM * tileN * sizeof(float));
        pipe.InitBuffer(tmpBuf, BUFFER_NUM, tileM * tileN * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t tile = this->startTile; tile < this->endTile; tile++) {
            uint32_t tileRowIdx = tile / this->tilesInN;
            uint32_t tileColIdx = tile % this->tilesInN;
            
            uint32_t mStart = tileRowIdx * this->tileM;
            uint32_t nStart = tileColIdx * this->tileN;
            
            uint32_t curM = this->tileM;
            if (mStart + curM > this->N) curM = this->N - mStart;
            uint32_t curN = this->tileN;
            if (nStart + curN > this->N) curN = this->N - nStart;

            // Initialize output tile to zero
            AscendC::LocalTensor<float> cLocal = outQueueC.AllocTensor<float>();
            AscendC::Duplicate(cLocal, (float)0, curM * curN);
            
            uint32_t kTiles = (this->N + this->tileK - 1) / this->tileK;
            
            for (uint32_t kt = 0; kt < kTiles; kt++) {
                uint32_t kStart = kt * this->tileK;
                uint32_t curK = this->tileK;
                if (kStart + curK > this->N) curK = this->N - kStart;
                
                // Load tile of A: rows [mStart, mStart+curM), cols [kStart, kStart+curK)
                // Since A is symmetric, A[i][j] = A[j][i], we can read A as-is
                AscendC::LocalTensor<float> aLocal = inQueueA.AllocTensor<float>();
                for (uint32_t i = 0; i < curM; i++) {
                    // Copy row (mStart+i), columns [kStart..kStart+curK)
                    uint32_t srcOffset = (mStart + i) * this->N + kStart;
                    uint32_t dstOffset = i * curK;
                    // Align copy length to 8 elements (32 bytes) for efficiency
                    uint32_t copyLen = curK;
                    // Use element-by-element if curK is small, otherwise DataCopy
                    if (copyLen >= 8) {
                        uint32_t alignedLen = (copyLen / 8) * 8;
                        AscendC::DataCopy(aLocal[dstOffset], aGm[srcOffset], alignedLen);
                        for (uint32_t r = alignedLen; r < copyLen; r++) {
                            float val;
                            AscendC::DataCopy(aLocal[dstOffset + r], aGm[srcOffset + r], 8);
                        }
                    } else {
                        AscendC::DataCopy(aLocal[dstOffset], aGm[srcOffset], 8);
                    }
                }
                inQueueA.EnQue(aLocal);
                
                // Load tile of B: rows [kStart, kStart+curK), cols [nStart, nStart+curN)
                AscendC::LocalTensor<float> bLocal = inQueueB.AllocTensor<float>();
                for (uint32_t i = 0; i < curK; i++) {
                    uint32_t srcOffset = (kStart + i) * this->N + nStart;
                    uint32_t dstOffset = i * curN;
                    if (curN >= 8) {
                        uint32_t alignedLen = (curN / 8) * 8;
                        AscendC::DataCopy(bLocal[dstOffset], bGm[srcOffset], alignedLen);
                    } else {
                        AscendC::DataCopy(bLocal[dstOffset], bGm[srcOffset], 8);
                    }
                }
                inQueueB.EnQue(bLocal);
                
                // Compute partial matmul
                aLocal = inQueueA.DeQue<float>();
                bLocal = inQueueB.DeQue<float>();
                
                AscendC::LocalTensor<float> tmpLocal = tmpBuf.AllocTensor<float>();
                
                for (uint32_t i = 0; i < curM; i++) {
                    for (uint32_t j = 0; j < curN; j++) {
                        float acc = 0.0f;
                        for (uint32_t k = 0; k < curK; k++) {
                            acc += aLocal.GetValue(i * curK + k) * bLocal.GetValue(k * curN + j);
                        }
                        tmpLocal.SetValue(i * curN + j, acc);
                    }
                }
                
                // Accumulate
                AscendC::Add(cLocal, cLocal, tmpLocal, curM * curN);
                
                tmpBuf.FreeTensor(tmpLocal);
                inQueueA.FreeTensor(aLocal);
                inQueueB.FreeTensor(bLocal);
            }
            
            // Write output tile back
            outQueueC.EnQue(cLocal);
            cLocal = outQueueC.DeQue<float>();
            for (uint32_t i = 0; i < curM; i++) {
                uint32_t dstOffset = (mStart + i) * this->N + nStart;
                uint32_t srcOffset = i * curN;
                if (curN >= 8) {
                    uint32_t alignedLen = (curN / 8) * 8;
                    AscendC::DataCopy(cGm[dstOffset], cLocal[srcOffset], alignedLen);
                } else {
                    AscendC::DataCopy(cGm[dstOffset], cLocal[srcOffset], 8);
                }
            }
            outQueueC.FreeTensor(cLocal);
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
    uint32_t N;
    uint32_t tileM, tileN, tileK;
    uint32_t tilesPerBlock;
    uint32_t startTile, endTile;
    uint32_t tilesInM, tilesInN;
};

extern "C" __global__ __aicore__ void matmul_for_symmetric_matrices_custom(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelMatmulSymmetric op;
    op.Init(a, b, c, tiling_data.N, tiling_data.tileM, tiling_data.tileN, tiling_data.tileK);
    op.Process();
}
