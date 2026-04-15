
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;

class KernelMatMul {
public:
    __aicore__ inline KernelMatMul() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t N, uint32_t tileM, uint32_t tileN, uint32_t tileK)
    {
        this->N = N;
        this->tileM = tileM;
        this->tileN = tileN;
        this->tileK = tileK;

        uint32_t totalBlocks = (N / tileM) * (N / tileN);
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t numBlocks = AscendC::GetBlockNum();

        this->startTask = blockIdx * ((totalBlocks + numBlocks - 1) / numBlocks);
        this->endTask = (blockIdx + 1) * ((totalBlocks + numBlocks - 1) / numBlocks);
        if (this->endTask > totalBlocks) this->endTask = totalBlocks;

        uint32_t tilesPerRow = N / tileN;
        (void)tilesPerRow;

        xGm.SetGlobalBuffer((__gm__ float *)x, N * N);
        yGm.SetGlobalBuffer((__gm__ float *)y, N * N);
        zGm.SetGlobalBuffer((__gm__ float *)z, N * N);

        pipe.InitBuffer(inQueueA, BUFFER_NUM, tileM * tileK * sizeof(float));
        pipe.InitBuffer(inQueueB, BUFFER_NUM, tileK * tileN * sizeof(float));
        pipe.InitBuffer(outQueueC, BUFFER_NUM, tileM * tileN * sizeof(float));
        pipe.InitBuffer(tmpBuf, 1, tileM * tileN * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        uint32_t tilesPerRow = N / tileN;
        uint32_t kTiles = N / tileK;

        for (uint32_t task = startTask; task < endTask; task++) {
            uint32_t tileRow = task / tilesPerRow;
            uint32_t tileCol = task % tilesPerRow;
            uint32_t rowOff = tileRow * tileM;
            uint32_t colOff = tileCol * tileN;

            // Zero out accumulator
            AscendC::LocalTensor<float> accLocal = outQueueC.AllocTensor<float>();
            AscendC::Duplicate(accLocal, (float)0, tileM * tileN);
            outQueueC.EnQue(accLocal);

            for (uint32_t kt = 0; kt < kTiles; kt++) {
                uint32_t kOff = kt * tileK;

                // Copy tile of A: rows [rowOff, rowOff+tileM), cols [kOff, kOff+tileK)
                AscendC::LocalTensor<float> aLocal = inQueueA.AllocTensor<float>();
                for (uint32_t i = 0; i < tileM; i++) {
                    AscendC::DataCopy(aLocal[i * tileK], xGm[(rowOff + i) * N + kOff], tileK);
                }
                inQueueA.EnQue(aLocal);

                // Copy tile of B: rows [kOff, kOff+tileK), cols [colOff, colOff+tileN)
                AscendC::LocalTensor<float> bLocal = inQueueB.AllocTensor<float>();
                for (uint32_t i = 0; i < tileK; i++) {
                    AscendC::DataCopy(bLocal[i * tileN], yGm[(kOff + i) * N + colOff], tileN);
                }
                inQueueB.EnQue(bLocal);

                // Compute partial matmul and accumulate
                aLocal = inQueueA.DeQue<float>();
                bLocal = inQueueB.DeQue<float>();
                AscendC::LocalTensor<float> cLocal = outQueueC.DeQue<float>();
                AscendC::LocalTensor<float> tempLocal = tmpBuf.AllocTensor<float>();

                // Simple matmul: C[i][j] += sum_k A[i][k] * B[k][j]
                for (uint32_t i = 0; i < tileM; i++) {
                    // For each row of A tile, multiply with all rows of B tile and accumulate
                    for (uint32_t k = 0; k < tileK; k++) {
                        // Broadcast A[i][k] and multiply with B[k][:]
                        AscendC::Muls(tempLocal, bLocal[k * tileN], aLocal[i * tileK + k], tileN);
                        AscendC::Add(cLocal[i * tileN], cLocal[i * tileN], tempLocal, tileN);
                    }
                }

                tmpBuf.FreeTensor(tempLocal);
                outQueueC.EnQue(cLocal);
                inQueueA.FreeTensor(aLocal);
                inQueueB.FreeTensor(bLocal);
            }

            // Copy result out
            AscendC::LocalTensor<float> cOut = outQueueC.DeQue<float>();
            for (uint32_t i = 0; i < tileM; i++) {
                AscendC::DataCopy(zGm[(rowOff + i) * N + colOff], cOut[i * tileN], tileN);
            }
            outQueueC.FreeTensor(cOut);
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueA, inQueueB;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueC;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    AscendC::GlobalTensor<float> zGm;
    uint32_t N;
    uint32_t tileM;
    uint32_t tileN;
    uint32_t tileK;
    uint32_t startTask;
    uint32_t endTask;
};

extern "C" __global__ __aicore__ void square_matrix_multiplication_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelMatMul op;
    op.Init(x, y, z, tiling_data.N, tiling_data.tileM, tiling_data.tileN, tiling_data.tileK);
    op.Process();
}
