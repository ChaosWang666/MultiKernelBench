
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelIndexCopy {
public:
    __aicore__ inline KernelIndexCopy() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR indices, GM_ADDR src, GM_ADDR z,
                                 uint32_t xRows, uint32_t cols, uint32_t srcRows)
    {
        this->xRows = xRows;
        this->cols = cols;
        this->srcRows = srcRows;
        uint32_t xTotal = xRows * cols;
        uint32_t srcTotal = srcRows * cols;

        xGm.SetGlobalBuffer((__gm__ float *)x, xTotal);
        indicesGm.SetGlobalBuffer((__gm__ int32_t *)indices, srcRows);
        srcGm.SetGlobalBuffer((__gm__ float *)src, srcTotal);
        zGm.SetGlobalBuffer((__gm__ float *)z, xTotal);

        // Align cols to 8 elements (32 bytes) for DataCopy
        uint32_t alignedCols = ((cols + 7) / 8) * 8;
        pipe.InitBuffer(inQueueX, BUFFER_NUM, alignedCols * sizeof(float));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, alignedCols * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        // First copy all x rows to z
        uint32_t alignedCols = ((cols + 7) / 8) * 8;
        for (uint32_t i = 0; i < xRows; i++) {
            AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
            AscendC::DataCopy(xLocal, xGm[i * cols], alignedCols);
            inQueueX.EnQue(xLocal);

            AscendC::LocalTensor<float> xOut = inQueueX.DeQue<float>();
            AscendC::DataCopy(zGm[i * cols], xOut, alignedCols);
            inQueueX.FreeTensor(xOut);
        }

        // Then for each index, copy src row to z at the indexed row
        for (uint32_t i = 0; i < srcRows; i++) {
            // Read index value from GM - we need to use DataCopy for indices too
            // Use a small aligned buffer for reading a single int32
            int32_t idx;
            // We'll read the index by copying a small block
            // For simplicity, read indices one at a time using DataCopy with alignment
            // Actually, we can use a workaround: copy aligned block of indices
            // But simpler: just read src row i and write to z[idx]
            
            // Read index: copy 8 int32 values (aligned) and pick the one we need
            // Actually let's just use a pipe buffer for indices
            // Simpler approach: batch read all indices first, then process

            // We'll use a direct approach with temporary buffer
            AscendC::LocalTensor<float> srcLocal = inQueueX.AllocTensor<float>();
            AscendC::DataCopy(srcLocal, srcGm[i * cols], alignedCols);
            inQueueX.EnQue(srcLocal);

            AscendC::LocalTensor<float> srcOut = inQueueX.DeQue<float>();

            // We need the index value. Since we can't easily do scalar reads,
            // we will copy indices in batches. For now let's handle it differently.
            // Use the global memory pointer cast approach
            int32_t idxVal = *((volatile __gm__ int32_t*)(indicesGm.GetPhyAddr() + i * sizeof(int32_t)));

            AscendC::DataCopy(zGm[idxVal * cols], srcOut, alignedCols);
            inQueueX.FreeTensor(srcOut);
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<int32_t> indicesGm;
    AscendC::GlobalTensor<float> srcGm;
    AscendC::GlobalTensor<float> zGm;
    uint32_t xRows;
    uint32_t cols;
    uint32_t srcRows;
};

extern "C" __global__ __aicore__ void index_copy_custom(GM_ADDR x, GM_ADDR indices, GM_ADDR src, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelIndexCopy op;
    op.Init(x, indices, src, z, tiling_data.xRows, tiling_data.cols, tiling_data.srcRows);
    op.Process();
}
