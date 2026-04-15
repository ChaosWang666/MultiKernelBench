
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;

class KernelMatmulUpperTri {
public:
    __aicore__ inline KernelMatmulUpperTri() {}
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, uint32_t N, uint32_t tileSize)
    {
        this->N = N;
        this->tileSize = tileSize;

        aGm.SetGlobalBuffer((__gm__ float *)a, N * N);
        bGm.SetGlobalBuffer((__gm__ float *)b, N * N);
        cGm.SetGlobalBuffer((__gm__ float *)c, N * N);

        // We need buffers for tile computation
        // aLocal for a tile of A row segment, bLocal for a tile of B column segment
        // accumLocal for accumulation, resultLocal for output
        pipe.InitBuffer(inQueueA, BUFFER_NUM, tileSize * sizeof(float));
        pipe.InitBuffer(inQueueB, BUFFER_NUM, tileSize * sizeof(float));
        pipe.InitBuffer(outQueue, BUFFER_NUM, tileSize * sizeof(float));
        pipe.InitBuffer(tmpBuf, BUFFER_NUM, tileSize * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        // Each block handles a subset of rows
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t rowsPerBlock = (N + blockNum - 1) / blockNum;
        uint32_t rowStart = blockIdx * rowsPerBlock;
        uint32_t rowEnd = rowStart + rowsPerBlock;
        if (rowEnd > N) rowEnd = N;

        for (uint32_t i = rowStart; i < rowEnd; i++) {
            // For upper triangular result, only compute columns j >= i
            for (uint32_t j = i; j < N; j++) {
                // C[i][j] = sum_{k=i}^{j} A[i][k] * B[k][j]
                // Because A is upper triangular (A[i][k]=0 for k<i)
                // and B is upper triangular (B[k][j]=0 for k>j)
                float sum = 0.0f;
                uint32_t kStart = i;
                uint32_t kEnd = j + 1; // k from i to j inclusive

                uint32_t kLen = kEnd - kStart;
                if (kLen == 0) continue;

                // Process in tiles
                uint32_t processed = 0;
                while (processed < kLen) {
                    uint32_t curLen = kLen - processed;
                    if (curLen > tileSize) curLen = tileSize;

                    // Align to 8 elements for DataCopy
                    uint32_t alignedLen = ((curLen + 7) / 8) * 8;
                    if (alignedLen > tileSize) alignedLen = tileSize;

                    AscendC::LocalTensor<float> aLocal = inQueueA.AllocTensor<float>();
                    AscendC::LocalTensor<float> bLocal = inQueueB.AllocTensor<float>();
                    AscendC::LocalTensor<float> tmpLocal = tmpBuf.AllocTensor<float>();

                    // Copy A[i][kStart+processed .. kStart+processed+alignedLen-1]
                    uint32_t aOffset = i * N + kStart + processed;
                    AscendC::DataCopy(aLocal, aGm[aOffset], alignedLen);

                    // Copy B[kStart+processed][j], B[kStart+processed+1][j], ...
                    // These are strided - column j of B starting at row kStart+processed
                    // We need to copy element by element or use strided copy
                    // For simplicity, set values individually
                    // Actually we need to gather column elements - let's do it with a loop per element
                    // and use scalar accumulation for correctness
                    
                    // Since strided gather is complex, do scalar computation
                    AscendC::SetMaskCount();
                    AscendC::SetMaskNorm();

                    // Zero out tmpLocal
                    AscendC::Duplicate(tmpLocal, (float)0.0f, alignedLen);
                    
                    // Gather B column values
                    for (uint32_t t = 0; t < curLen; t++) {
                        uint32_t bOffset = (kStart + processed + t) * N + j;
                        // We'll copy a single aligned block containing our value
                        // For simplicity, just accumulate scalarly
                        tmpLocal.SetValue(t, bGm.GetValue(bOffset));
                    }

                    // Multiply element-wise and reduce
                    AscendC::LocalTensor<float> resLocal = outQueue.AllocTensor<float>();
                    AscendC::Mul(resLocal, aLocal, tmpLocal, alignedLen);

                    // Sum reduction
                    for (uint32_t t = 0; t < curLen; t++) {
                        sum += resLocal.GetValue(t);
                    }

                    outQueue.FreeTensor(resLocal);
                    inQueueA.FreeTensor(aLocal);
                    inQueueB.FreeTensor(bLocal);
                    tmpBuf.FreeTensor(tmpLocal);

                    processed += curLen;
                }

                cGm.SetValue(i * N + j, sum);
            }
            // Set lower triangular part to zero for this row
            for (uint32_t j = 0; j < i; j++) {
                cGm.SetValue(i * N + j, 0.0f);
            }
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueA, inQueueB;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueue;
    AscendC::TQue<AscendC::TPosition::VECCALC, BUFFER_NUM> tmpBuf;
    AscendC::GlobalTensor<float> aGm;
    AscendC::GlobalTensor<float> bGm;
    AscendC::GlobalTensor<float> cGm;
    uint32_t N;
    uint32_t tileSize;
};

extern "C" __global__ __aicore__ void matmul_for_upper_triangular_matrices_custom(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelMatmulUpperTri op;
    op.Init(a, b, c, tiling_data.N, tiling_data.tileSize);
    op.Process();
}
