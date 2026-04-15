
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;

class KernelMatmulTril {
public:
    __aicore__ inline KernelMatmulTril() {}
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, uint32_t N, uint32_t tileNum)
    {
        this->N = N;
        this->tileNum = tileNum;

        uint32_t totalRows = N;
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();

        this->rowsPerBlock = (totalRows + blockNum - 1) / blockNum;
        this->startRow = blockIdx * this->rowsPerBlock;
        this->endRow = startRow + rowsPerBlock;
        if (this->endRow > N) this->endRow = N;
        if (this->startRow >= N) {
            this->startRow = N;
            this->endRow = N;
        }

        aGm.SetGlobalBuffer((__gm__ float *)a, N * N);
        bGm.SetGlobalBuffer((__gm__ float *)b, N * N);
        cGm.SetGlobalBuffer((__gm__ float *)c, N * N);

        // We need buffers for loading a row of A, a column of B, and accumulation
        // tile size for dot product computation
        this->tileLen = ((N + 7) / 8) * 8; // align to 8 floats (32 bytes)

        pipe.InitBuffer(inQueueA, BUFFER_NUM, this->tileLen * sizeof(float));
        pipe.InitBuffer(inQueueB, BUFFER_NUM, this->tileLen * sizeof(float));
        pipe.InitBuffer(outQueueC, BUFFER_NUM, this->tileLen * sizeof(float));
        pipe.InitBuffer(tmpBuf, BUFFER_NUM, this->tileLen * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = this->startRow; i < this->endRow; i++) {
            ComputeRow(i);
        }
    }

private:
    __aicore__ inline void ComputeRow(uint32_t row)
    {
        // For row i of C (lower triangular result), C[i][j] = 0 for j > i
        // C[i][j] = sum_{k=j}^{i} A[i][k] * B[k][j] for j <= i
        // We compute all columns j=0..i together by loading row i of A and
        // for each k, multiplying A[i][k] by row k of B, accumulating into C row.

        // Allocate output buffer and zero it
        AscendC::LocalTensor<float> cLocal = outQueueC.AllocTensor<float>();

        // Zero out cLocal
        AscendC::Duplicate<float>(cLocal, 0.0f, this->tileLen);

        for (uint32_t k = 0; k <= row; k++) {
            // Load A[row][k] - single element
            float aVal = 0.0f;

            // Load row k of B into bLocal (only first k+1 elements are non-zero for lower tri)
            AscendC::LocalTensor<float> bLocal = inQueueB.AllocTensor<float>();

            // Zero bLocal first
            AscendC::Duplicate<float>(bLocal, 0.0f, this->tileLen);

            // Copy row k of B - we only need columns 0..row (since C[row][j]=0 for j>row)
            uint32_t copyLen = row + 1;
            uint32_t copyLenAligned = ((copyLen + 7) / 8) * 8;

            AscendC::DataCopy(bLocal, bGm[k * N], copyLenAligned);

            // Load scalar A[row][k]
            AscendC::LocalTensor<float> aLocal = inQueueA.AllocTensor<float>();
            // We load a small aligned chunk containing A[row][k]
            uint32_t aOffset = row * N + k;
            uint32_t aAlignedStart = (aOffset / 8) * 8;
            AscendC::DataCopy(aLocal, aGm[aAlignedStart], 8);
            uint32_t aLocalIdx = aOffset - aAlignedStart;
            aVal = aLocal.GetValue(aLocalIdx);
            inQueueA.FreeTensor(aLocal);

            // cLocal += aVal * bLocal
            AscendC::LocalTensor<float> tLocal = tmpBuf.AllocTensor<float>();
            AscendC::Muls(tLocal, bLocal, aVal, this->tileLen);
            AscendC::Add(cLocal, cLocal, tLocal, this->tileLen);
            tmpBuf.FreeTensor(tLocal);

            inQueueB.FreeTensor(bLocal);
        }

        // Zero out columns > row (enforce lower triangular)
        for (uint32_t j = row + 1; j < this->tileLen; j++) {
            cLocal.SetValue(j, 0.0f);
        }

        // Write back row
        uint32_t writeLenAligned = ((N + 7) / 8) * 8;
        AscendC::DataCopy(cGm[row * N], cLocal, writeLenAligned);

        outQueueC.FreeTensor(cLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueA, inQueueB;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueC;
    AscendC::TQue<AscendC::TPosition::VECCALC, BUFFER_NUM> tmpBuf;
    AscendC::GlobalTensor<float> aGm;
    AscendC::GlobalTensor<float> bGm;
    AscendC::GlobalTensor<float> cGm;
    uint32_t N;
    uint32_t tileNum;
    uint32_t tileLen;
    uint32_t rowsPerBlock;
    uint32_t startRow;
    uint32_t endRow;
};

extern "C" __global__ __aicore__ void matmul_for_lower_triangular_matrices_custom(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelMatmulTril op;
    op.Init(a, b, c, tiling_data.N, tiling_data.tileNum);
    op.Process();
}
