
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;

class KernelFourDimTensorMatMul {
public:
    __aicore__ inline KernelFourDimTensorMatMul() {}
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c,
                                 uint32_t batchSize, uint32_t M, uint32_t K, uint32_t N)
    {
        this->batchSize = batchSize;
        this->M = M;
        this->K = K;
        this->N = N;

        uint32_t totalRows = batchSize * M;
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();

        this->rowsPerBlock = totalRows / blockNum;
        uint32_t remainRows = totalRows % blockNum;
        if (blockIdx < remainRows) {
            this->rowsPerBlock += 1;
            this->startRow = blockIdx * this->rowsPerBlock;
        } else {
            this->startRow = blockIdx * this->rowsPerBlock + remainRows;
        }

        aGm.SetGlobalBuffer((__gm__ float*)a, batchSize * M * K);
        bGm.SetGlobalBuffer((__gm__ float*)b, K * N);
        cGm.SetGlobalBuffer((__gm__ float*)c, batchSize * M * N);

        // Determine tile size for K dimension based on available UB
        // Each tile processes a chunk of K for one row of A and produces partial results
        // We need: tileK floats for A row chunk, N floats for one row of B (but we iterate), tileK*some for B chunk, N floats for accumulator
        // Simpler approach: process one row at a time, tile along K
        // For each row: load A[row, 0:K], and for each k, accumulate A[row,k]*B[k,0:N] into C[row,0:N]
        // But this is very slow element-by-element.
        
        // Better: tile along N dimension for output, and process full K reduction
        // For a tile of N columns: load A_row[0:K], load B[0:K, n_start:n_start+tileN], compute dot products
        
        // Simplest viable approach for AscendC: 
        // Process one output row at a time
        // For each row, tile along N
        // For each N-tile, iterate over K and accumulate
        
        // UB budget: ~256KB = 65536 floats
        // We need: tileN floats for accumulator, tileK floats for A chunk, tileK * tileN for B chunk? Too much.
        // Alternative: iterate over K one element at a time (scalar), broadcast multiply
        // Load A[row, k] as scalar, load B[k, n_start:n_start+tileN], multiply, accumulate
        
        // tileN for B row: tileN floats
        // tileN for accumulator: tileN floats  
        // 1 float for A element (but we can load a chunk of A)
        // tileN can be large, say up to 8192
        
        this->tileN = N;
        if (this->tileN > 8192) {
            this->tileN = 8192;
        }
        // Align tileN to 8 (32 bytes / 4 bytes per float)
        this->tileN = (this->tileN + 7) / 8 * 8;
        if (this->tileN > N) {
            this->tileN = N;
            this->tileN = (this->tileN + 7) / 8 * 8;
        }

        // Buffer sizes
        // We need: accum buffer (tileN), B row buffer (tileN), A chunk buffer (some K chunk), scalar broadcast buffer (tileN)
        uint32_t alignedTileN = this->tileN;
        
        pipe.InitBuffer(accumBuf, 1, alignedTileN * sizeof(float));
        pipe.InitBuffer(bRowBuf, 1, alignedTileN * sizeof(float));
        pipe.InitBuffer(scalarBuf, 1, alignedTileN * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t r = 0; r < this->rowsPerBlock; r++) {
            uint32_t globalRow = this->startRow + r;
            if (globalRow >= this->batchSize * this->M) break;
            ProcessRow(globalRow);
        }
    }

private:
    __aicore__ inline void ProcessRow(uint32_t globalRow)
    {
        uint32_t aRowOffset = globalRow * this->K;
        uint32_t cRowOffset = globalRow * this->N;

        for (uint32_t nStart = 0; nStart < this->N; nStart += this->tileN) {
            uint32_t curTileN = this->tileN;
            if (nStart + curTileN > this->N) {
                curTileN = this->N - nStart;
            }
            uint32_t alignedCurTileN = (curTileN + 7) / 8 * 8;

            AscendC::LocalTensor<float> accum = accumBuf.Get<float>();
            AscendC::LocalTensor<float> bRow = bRowBuf.Get<float>();
            AscendC::LocalTensor<float> scalarTensor = scalarBuf.Get<float>();

            // Zero out accumulator
            AscendC::Duplicate<float>(accum, 0.0f, alignedCurTileN);

            for (uint32_t kk = 0; kk < this->K; kk++) {
                // Load A[globalRow, kk]
                float aVal = *((__gm__ float*)(aGm.GetPhyAddr()) + aRowOffset + kk);
                
                // Load B[kk, nStart:nStart+curTileN]
                AscendC::DataCopy(bRow, bGm[kk * this->N + nStart], alignedCurTileN);
                AscendC::PipeBarrier<PIPE_ALL>();

                // Broadcast aVal and multiply
                AscendC::Duplicate<float>(scalarTensor, aVal, alignedCurTileN);
                AscendC::PipeBarrier<PIPE_ALL>();

                // accum += aVal * bRow
                AscendC::Muls(bRow, bRow, aVal, alignedCurTileN);
                AscendC::PipeBarrier<PIPE_ALL>();
                AscendC::Add(accum, accum, bRow, alignedCurTileN);
                AscendC::PipeBarrier<PIPE_ALL>();
            }

            // Write result
            AscendC::DataCopy(cGm[cRowOffset + nStart], accum, alignedCurTileN);
            AscendC::PipeBarrier<PIPE_ALL>();
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> accumBuf, bRowBuf, scalarBuf;
    AscendC::GlobalTensor<float> aGm;
    AscendC::GlobalTensor<float> bGm;
    AscendC::GlobalTensor<float> cGm;
    uint32_t batchSize, M, K, N;
    uint32_t startRow, rowsPerBlock;
    uint32_t tileN;
};

extern "C" __global__ __aicore__ void four_dim_tensor_matrix_multiplication_custom(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelFourDimTensorMatMul op;
    op.Init(a, b, c, tiling_data.batchSize, tiling_data.M, tiling_data.K, tiling_data.N);
    op.Process();
}
