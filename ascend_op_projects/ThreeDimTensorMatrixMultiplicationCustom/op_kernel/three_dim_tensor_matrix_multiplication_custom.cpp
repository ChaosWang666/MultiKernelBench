
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;

class KernelMatMul3D {
public:
    __aicore__ inline KernelMatMul3D() {}
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, uint32_t N, uint32_t M, uint32_t K, uint32_t L)
    {
        this->N = N;
        this->M = M;
        this->K = K;
        this->L = L;
        
        uint32_t totalRows = N * M;
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        
        this->rowsPerBlock = totalRows / blockNum;
        uint32_t remainder = totalRows % blockNum;
        
        if (blockIdx < remainder) {
            this->rowsPerBlock += 1;
            this->startRow = blockIdx * this->rowsPerBlock;
        } else {
            this->startRow = blockIdx * this->rowsPerBlock + remainder;
        }
        
        aGm.SetGlobalBuffer((__gm__ float *)a, N * M * K);
        bGm.SetGlobalBuffer((__gm__ float *)b, K * L);
        cGm.SetGlobalBuffer((__gm__ float *)c, N * M * L);
        
        // We process tiles of K dimension. Determine tile size for K.
        // Each tile of A row: tileK floats, each tile of B: tileK * L floats
        // We need to fit in UB. Let's use a reasonable tile size.
        // UB is typically 256KB = 65536 floats
        // We need: tileK (for A tile) + tileK * tileL (for B tile) + tileL (for C accum) + tileL (for partial)
        // Let's tile both K and L dimensions
        
        // Choose tileL to align to 8 floats (32 bytes)
        // Choose tileK to align to 8 floats
        this->tileL = L;
        if (this->tileL > 256) {
            this->tileL = 256;
        }
        // Align tileL to 8
        this->tileL = (this->tileL + 7) / 8 * 8;
        if (this->tileL > L) {
            this->tileL = (L + 7) / 8 * 8;
        }
        
        // tileK: we need tileK + tileK * tileL + tileL + tileL floats in UB
        // Budget ~ 49152 floats (192KB to be safe)
        // tileK * (1 + tileL) + 2 * tileL <= 49152
        // tileK <= (49152 - 2*tileL) / (1 + tileL)
        uint32_t budget = 32768;
        this->tileK = (budget - 2 * this->tileL) / (1 + this->tileL);
        if (this->tileK > K) {
            this->tileK = K;
        }
        // Align tileK to 8
        this->tileK = (this->tileK / 8) * 8;
        if (this->tileK < 8) this->tileK = 8;
        
        pipe.InitBuffer(aBuf, BUFFER_NUM, this->tileK * sizeof(float));
        pipe.InitBuffer(bBuf, BUFFER_NUM, this->tileK * this->tileL * sizeof(float));
        pipe.InitBuffer(cBuf, BUFFER_NUM, this->tileL * sizeof(float));
        pipe.InitBuffer(tmpBuf, BUFFER_NUM, this->tileL * sizeof(float));
    }
    
    __aicore__ inline void Process()
    {
        for (uint32_t row = 0; row < this->rowsPerBlock; row++) {
            uint32_t globalRow = this->startRow + row;
            if (globalRow >= N * M) break;
            
            // Process each tile of L
            for (uint32_t lStart = 0; lStart < L; lStart += this->tileL) {
                uint32_t curTileL = this->tileL;
                if (lStart + curTileL > L) {
                    curTileL = L - lStart;
                }
                uint32_t alignedTileL = (curTileL + 7) / 8 * 8;
                
                // Initialize accumulator to zero
                AscendC::LocalTensor<float> cLocal = cBuf.AllocTensor<float>();
                AscendC::Duplicate(cLocal, (float)0.0f, alignedTileL);
                
                // Process tiles of K
                for (uint32_t kStart = 0; kStart < K; kStart += this->tileK) {
                    uint32_t curTileK = this->tileK;
                    if (kStart + curTileK > K) {
                        curTileK = K - kStart;
                    }
                    uint32_t alignedTileK = (curTileK + 7) / 8 * 8;
                    
                    // Load A tile: A[globalRow, kStart:kStart+curTileK]
                    AscendC::LocalTensor<float> aLocal = aBuf.AllocTensor<float>();
                    AscendC::DataCopy(aLocal, aGm[globalRow * K + kStart], alignedTileK);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
                    
                    // Load B tile: B[kStart:kStart+curTileK, lStart:lStart+curTileL]
                    // B is (K, L), we need curTileK rows, each of curTileL elements starting at col lStart
                    AscendC::LocalTensor<float> bLocal = bBuf.AllocTensor<float>();
                    
                    // Use DataCopy with stride parameters for non-contiguous B
                    // Each row of B has L elements, we want curTileL elements per row
                    if (curTileL == L && alignedTileL == L) {
                        // Contiguous copy
                        AscendC::DataCopy(bLocal, bGm[(kStart) * L + lStart], alignedTileK * alignedTileL);
                    } else {
                        // Strided copy: curTileK blocks, each of alignedTileL elements
                        // srcStride = L elements between rows, dstStride = alignedTileL
                        AscendC::DataCopyParams copyParams;
                        copyParams.blockCount = curTileK;
                        copyParams.blockLen = (uint16_t)(alignedTileL * sizeof(float) / 32);
                        copyParams.srcStride = (uint16_t)((L - alignedTileL) * sizeof(float) / 32);
                        copyParams.dstStride = 0;
                        AscendC::DataCopy(bLocal, bGm[kStart * L + lStart], copyParams);
                    }
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(1);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(1);
                    
                    // Compute: for each k in curTileK, c[l] += a[k] * b[k, l]
                    AscendC::LocalTensor<float> tmpLocal = tmpBuf.AllocTensor<float>();
                    for (uint32_t ki = 0; ki < curTileK; ki++) {
                        // Multiply: tmp = a[ki] * B[ki, :]
                        AscendC::Muls(tmpLocal, bLocal[ki * alignedTileL], aLocal.GetValue(ki), alignedTileL);
                        // Accumulate: c += tmp
                        AscendC::Add(cLocal, cLocal, tmpLocal, alignedTileL);
                    }
                    tmpBuf.FreeTensor(tmpLocal);
                    
                    aBuf.FreeTensor(aLocal);
                    bBuf.FreeTensor(bLocal);
                }
                
                // Write C tile back: C[globalRow, lStart:lStart+curTileL]
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
                if (curTileL == alignedTileL) {
                    AscendC::DataCopy(cGm[globalRow * L + lStart], cLocal, alignedTileL);
                } else {
                    AscendC::DataCopyParams outParams;
                    outParams.blockCount = 1;
                    outParams.blockLen = (uint16_t)((curTileL * sizeof(float) + 31) / 32);
                    outParams.srcStride = 0;
                    outParams.dstStride = 0;
                    AscendC::DataCopy(cGm[globalRow * L + lStart], cLocal, outParams);
                }
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
                
                cBuf.FreeTensor(cLocal);
            }
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> aBuf, bBuf, cBuf, tmpBuf;
    AscendC::GlobalTensor<float> aGm;
    AscendC::GlobalTensor<float> bGm;
    AscendC::GlobalTensor<float> cGm;
    uint32_t N, M, K, L;
    uint32_t startRow, rowsPerBlock;
    uint32_t tileK, tileL;
};

extern "C" __global__ __aicore__ void three_dim_tensor_matrix_multiplication_custom(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelMatMul3D op;
    op.Init(a, b, c, tiling_data.N, tiling_data.M, tiling_data.K, tiling_data.L);
    op.Process();
}
