
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;

class KernelMatmul {
public:
    __aicore__ inline KernelMatmul() {}
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, uint32_t M, uint32_t K, uint32_t N)
    {
        this->M = M;
        this->K = K;
        this->N = N;

        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();

        // Each block handles a chunk of rows of the output
        this->rowsPerBlock = (M + blockNum - 1) / blockNum;
        this->startRow = blockIdx * this->rowsPerBlock;
        this->endRow = startRow + rowsPerBlock;
        if (this->endRow > M) this->endRow = M;
        if (this->startRow >= M) {
            this->startRow = M;
            this->endRow = M;
        }
        this->myRows = this->endRow - this->startRow;

        aGm.SetGlobalBuffer((__gm__ float *)a, M * K);
        bGm.SetGlobalBuffer((__gm__ float *)b, K * N);
        cGm.SetGlobalBuffer((__gm__ float *)c, M * N);

        // We'll process tile by tile. Determine tile sizes based on available UB.
        // Each tile: TILE_M rows x TILE_N cols of C, requiring TILE_M x TILE_K of A and TILE_K x TILE_N of B
        // UB size is limited. We'll use conservative tile sizes.
        this->TILE_M = 16;
        this->TILE_N = 128;
        this->TILE_K = 64;

        // Allocate buffers for tiles
        uint32_t aSize = TILE_M * TILE_K;
        uint32_t bSize = TILE_K * TILE_N;
        uint32_t cSize = TILE_M * TILE_N;
        uint32_t tempSize = TILE_M * TILE_N;

        pipe.InitBuffer(inQueueA, BUFFER_NUM, aSize * sizeof(float));
        pipe.InitBuffer(inQueueB, BUFFER_NUM, bSize * sizeof(float));
        pipe.InitBuffer(outQueueC, BUFFER_NUM, cSize * sizeof(float));
        pipe.InitBuffer(tmpBuf, BUFFER_NUM, tempSize * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (myRows == 0) return;

        // Iterate over output tile blocks
        for (uint32_t mStart = 0; mStart < myRows; mStart += TILE_M) {
            uint32_t curTileM = TILE_M;
            if (mStart + curTileM > myRows) curTileM = myRows - mStart;

            for (uint32_t nStart = 0; nStart < N; nStart += TILE_N) {
                uint32_t curTileN = TILE_N;
                if (nStart + curTileN > N) curTileN = N - nStart;

                // Initialize accumulator to zero
                AscendC::LocalTensor<float> cLocal = outQueueC.AllocTensor<float>();
                AscendC::Duplicate(cLocal, (float)0, curTileM * curTileN);

                // Accumulate over K dimension
                for (uint32_t kStart = 0; kStart < K; kStart += TILE_K) {
                    uint32_t curTileK = TILE_K;
                    if (kStart + curTileK > K) curTileK = K - kStart;

                    // Load A tile: curTileM x curTileK
                    AscendC::LocalTensor<float> aLocal = inQueueA.AllocTensor<float>();
                    for (uint32_t i = 0; i < curTileM; i++) {
                        uint32_t globalRow = this->startRow + mStart + i;
                        AscendC::DataCopy(aLocal[i * TILE_K], aGm[globalRow * K + kStart], curTileK);
                    }
                    inQueueA.EnQue(aLocal);

                    // Load B tile: curTileK x curTileN
                    AscendC::LocalTensor<float> bLocal = inQueueB.AllocTensor<float>();
                    for (uint32_t i = 0; i < curTileK; i++) {
                        AscendC::DataCopy(bLocal[i * TILE_N], bGm[(kStart + i) * N + nStart], curTileN);
                    }
                    inQueueB.EnQue(bLocal);

                    // Compute partial matmul and accumulate
                    AscendC::LocalTensor<float> aComp = inQueueA.DeQue<float>();
                    AscendC::LocalTensor<float> bComp = inQueueB.DeQue<float>();
                    AscendC::LocalTensor<float> tmpLocal = tmpBuf.AllocTensor<float>();

                    for (uint32_t i = 0; i < curTileM; i++) {
                        for (uint32_t kk = 0; kk < curTileK; kk++) {
                            float aVal = aComp.GetValue(i * TILE_K + kk);
                            // tmpLocal = aVal * bComp[kk * TILE_N ... kk * TILE_N + curTileN]
                            AscendC::Muls(tmpLocal, bComp[kk * TILE_N], aVal, curTileN);
                            AscendC::Add(cLocal[i * TILE_N], cLocal[i * TILE_N], tmpLocal, curTileN);
                        }
                    }

                    tmpBuf.FreeTensor(tmpLocal);
                    inQueueA.FreeTensor(aComp);
                    inQueueB.FreeTensor(bComp);
                }

                // Store C tile
                outQueueC.EnQue(cLocal);
                AscendC::LocalTensor<float> cOut = outQueueC.DeQue<float>();
                for (uint32_t i = 0; i < curTileM; i++) {
                    uint32_t globalRow = this->startRow + mStart + i;
                    AscendC::DataCopy(cGm[globalRow * N + nStart], cOut[i * TILE_N], curTileN);
                }
                outQueueC.FreeTensor(cOut);
            }
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueA, inQueueB;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueC;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    AscendC::GlobalTensor<float> aGm, bGm, cGm;
    uint32_t M, K, N;
    uint32_t rowsPerBlock, startRow, endRow, myRows;
    uint32_t TILE_M, TILE_N, TILE_K;
};

extern "C" __global__ __aicore__ void standard_matrix_multiplication_custom(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelMatmul op;
    op.Init(a, b, c, tiling_data.M, tiling_data.K, tiling_data.N);
    op.Process();
}
