
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelMatmulDiag {
public:
    __aicore__ inline KernelMatmulDiag() {}
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, uint32_t N, uint32_t M, uint32_t tileNum)
    {
        this->N = N;
        this->M = M;
        this->totalRows = N;
        // Each block processes a chunk of rows
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        this->rowsPerBlock = (N + blockNum - 1) / blockNum;
        this->startRow = blockIdx * this->rowsPerBlock;
        if (this->startRow + this->rowsPerBlock > N) {
            this->rowsPerBlock = (this->startRow < N) ? (N - this->startRow) : 0;
        }
        this->tileNum = tileNum;
        // Each row has M elements; we tile along M dimension
        // tileLength is number of float elements per tile along M
        this->tileLength = (M + tileNum * BUFFER_NUM - 1) / (tileNum * BUFFER_NUM);
        // Align tileLength to 8 (32 bytes / 4 bytes per float)
        this->tileLength = (this->tileLength + 7) / 8 * 8;

        aGm.SetGlobalBuffer((__gm__ float *)a, N);
        bGm.SetGlobalBuffer((__gm__ float *)b, N * M);
        cGm.SetGlobalBuffer((__gm__ float *)c, N * M);

        pipe.InitBuffer(inQueueB, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueC, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(inQueueA, 1, this->tileLength * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        if (this->rowsPerBlock == 0) return;
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (uint32_t row = 0; row < this->rowsPerBlock; row++) {
            uint32_t globalRow = this->startRow + row;
            if (globalRow >= this->N) break;
            // Load the diagonal element for this row
            float diagVal;
            // We read one element from aGm
            // Process tiles along M dimension
            for (int32_t i = 0; i < loopCount; i++) {
                uint32_t offset = i * this->tileLength;
                if (offset >= this->M) break;
                uint32_t curLen = this->tileLength;
                if (offset + curLen > this->M) {
                    curLen = this->M - offset;
                    // Align to 8
                    curLen = (curLen + 7) / 8 * 8;
                }
                CopyInB(globalRow, offset, curLen);
                Compute(globalRow, curLen);
                CopyOut(globalRow, offset, curLen);
            }
        }
    }

private:
    __aicore__ inline void CopyInB(uint32_t row, uint32_t colOffset, uint32_t len)
    {
        AscendC::LocalTensor<float> bLocal = inQueueB.AllocTensor<float>();
        AscendC::DataCopy(bLocal, bGm[row * this->M + colOffset], len);
        inQueueB.EnQue(bLocal);
    }
    __aicore__ inline void Compute(uint32_t row, uint32_t len)
    {
        AscendC::LocalTensor<float> bLocal = inQueueB.DeQue<float>();
        AscendC::LocalTensor<float> cLocal = outQueueC.AllocTensor<float>();
        AscendC::LocalTensor<float> aLocal = inQueueA.AllocTensor<float>();
        // Duplicate diagonal value across the tile
        AscendC::DataCopy(aLocal, aGm[row], 8);
        // aLocal[0] now has the diagonal value
        float diagVal = aLocal.GetValue(0);
        AscendC::Muls(cLocal, bLocal, diagVal, len);
        outQueueC.EnQue(cLocal);
        inQueueB.FreeTensor(bLocal);
        inQueueA.FreeTensor(aLocal);
    }
    __aicore__ inline void CopyOut(uint32_t row, uint32_t colOffset, uint32_t len)
    {
        AscendC::LocalTensor<float> cLocal = outQueueC.DeQue<float>();
        uint32_t actualLen = len;
        if (colOffset + actualLen > this->M) {
            actualLen = this->M - colOffset;
            actualLen = (actualLen + 7) / 8 * 8;
        }
        AscendC::DataCopy(cGm[row * this->M + colOffset], cLocal, actualLen);
        outQueueC.FreeTensor(cLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueB;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueA;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueC;
    AscendC::GlobalTensor<float> aGm;
    AscendC::GlobalTensor<float> bGm;
    AscendC::GlobalTensor<float> cGm;
    uint32_t N;
    uint32_t M;
    uint32_t totalRows;
    uint32_t rowsPerBlock;
    uint32_t startRow;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void matmul_with_diagonal_matrices_custom(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelMatmulDiag op;
    op.Init(a, b, c, tiling_data.N, tiling_data.M, tiling_data.tileNum);
    op.Process();
}
