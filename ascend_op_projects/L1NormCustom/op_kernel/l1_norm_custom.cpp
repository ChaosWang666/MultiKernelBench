
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelL1Norm {
public:
    __aicore__ inline KernelL1Norm() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t batchSize, uint32_t dim, uint32_t tileNum)
    {
        this->batchSize = batchSize;
        this->dim = dim;
        this->tileNum = tileNum;

        // Each block processes a subset of rows
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        this->rowsPerBlock = (batchSize + blockNum - 1) / blockNum;
        this->startRow = blockIdx * this->rowsPerBlock;
        this->endRow = startRow + rowsPerBlock;
        if (this->endRow > batchSize) {
            this->endRow = batchSize;
        }

        // Tile length for processing dim
        this->tileLength = (dim + tileNum - 1) / tileNum;
        // Align to 8 floats (32 bytes)
        this->tileLength = ((this->tileLength + 7) / 8) * 8;

        xGm.SetGlobalBuffer((__gm__ float *)x, batchSize * (uint64_t)dim);
        yGm.SetGlobalBuffer((__gm__ float *)y, batchSize * (uint64_t)dim);

        // Allocate buffers
        // We need: inBuf for loading a tile, tmpBuf for abs, sumBuf for partial sum
        pipe.InitBuffer(inQueue, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueue, BUFFER_NUM, this->tileLength * sizeof(float));
        // For the reduction sum, we use a work buffer
        pipe.InitBuffer(workBuf, 1, this->tileLength * sizeof(float));
        // For scalar broadcasting
        pipe.InitBuffer(scalarBuf, 1, this->tileLength * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t row = this->startRow; row < this->endRow; row++) {
            ProcessRow(row);
        }
    }

private:
    __aicore__ inline void ProcessRow(uint32_t row)
    {
        uint64_t rowOffset = row * (uint64_t)this->dim;

        // First pass: compute sum of abs values
        AscendC::LocalTensor<float> sumLocal = workBuf.Get<float>();
        // Initialize sum to 0
        AscendC::Duplicate(sumLocal, (float)0.0f, this->tileLength);

        uint32_t remaining = this->dim;
        uint32_t offset = 0;
        for (uint32_t t = 0; t < this->tileNum; t++) {
            if (remaining == 0) break;
            uint32_t curLen = this->tileLength;
            if (curLen > remaining) {
                curLen = remaining;
            }
            // Align curLen up to 8 for DataCopy
            uint32_t copyLen = ((curLen + 7) / 8) * 8;
            if (copyLen > this->tileLength) copyLen = this->tileLength;

            AscendC::LocalTensor<float> inLocal = inQueue.AllocTensor<float>();
            // Zero out the buffer first if copyLen > curLen to avoid garbage in padding
            if (copyLen > curLen) {
                AscendC::Duplicate(inLocal, (float)0.0f, copyLen);
            }
            AscendC::DataCopy(inLocal, xGm[rowOffset + offset], copyLen);
            inQueue.EnQue(inLocal);

            AscendC::LocalTensor<float> inLocal2 = inQueue.DeQue<float>();
            // Compute abs
            AscendC::Abs(inLocal2, inLocal2, copyLen);
            // Add to sum
            AscendC::Add(sumLocal, sumLocal, inLocal2, copyLen);
            inQueue.FreeTensor(inLocal2);

            offset += curLen;
            remaining -= curLen;
        }

        // Now reduce sumLocal to a scalar
        // sumLocal has tileLength elements, we need to sum them all
        // Use ReduceSum
        float totalSum = 0.0f;
        // We'll do a reduction over the accumulated buffer
        // The sumLocal contains partial sums across tiles, but they are element-wise accumulated
        // Actually we need the total sum of all elements. Let's use AscendC::ReduceSum
        AscendC::LocalTensor<float> scalarLocal = scalarBuf.Get<float>();
        
        // ReduceSum: reduce tileLength elements to 1
        // Work tensor needs tileLength elements
        // Actually we can just sum using a loop-based reduction or use the built-in
        // Let's use a simpler approach: WholeReduceSum
        float rowSum = 0.0f;
        // Use block reduce approach
        AscendC::LocalTensor<float> reduceResult = scalarLocal;
        uint32_t reduceLen = this->tileLength;
        // ReduceSum requires length to be power-of-2 friendly, let's do manual
        // Actually AscendC::ReduceSum should work on aligned lengths
        // Use the pattern: ReduceSum(dst, src, workLocal, reduceLen)
        // dst should have at least reduceLen/8 elements as workspace
        // This is complicated, let's just do a simpler approach
        
        // Copy sumLocal to a temp and repeatedly halve
        // Or just use WholeReduceSum if available
        // Let's try: sum all tileLength elements
        // Since tileLength may be large, use AscendC vector reduce
        
        // Simple approach: repeatedly add halves
        AscendC::Duplicate(reduceResult, (float)0.0f, this->tileLength);
        AscendC::DataCopy(reduceResult, sumLocal, this->tileLength);
        
        uint32_t len = this->tileLength;
        while (len > 1) {
            uint32_t half = len / 2;
            if (half == 0) break;
            // Align half to 8
            uint32_t alignedHalf = ((half + 7) / 8) * 8;
            // Add second half to first half
            // We need a view of the second half
            AscendC::Add(reduceResult, reduceResult, reduceResult[half], alignedHalf);
            len = half;
        }
        
        // Now reduceResult[0] has the sum
        // Compute mean = totalSum / dim
        float meanVal;
        // We can't easily read a single scalar, let's use Duplicate + Div approach
        // Actually let's compute 1/mean and multiply
        
        // Compute reciprocal of (sum / dim) = dim / sum
        // Use Muls to scale: result[i] = x[i] * dim / sum = x[i] / mean
        // But we need the scalar sum first
        
        // Use a different strategy: store sum as a vector of same value, then divide
        // AscendC::Muls(dst, src, scalar, len) - multiply by scalar
        // But we need the scalar value...
        
        // Let's just use Div: y = x / meanTensor where meanTensor is broadcast
        // First, let's get the sum into a scalar by using Muls with 1/dim
        // Then broadcast it
        
        // Actually, a cleaner approach:
        // After reduction, reduceResult[0] = sum of abs values
        // We want to divide each element by (sum / dim) = multiply by (dim / sum)
        // Use AscendC::Divs to divide reduceResult[0] by itself to get 1, then scale... no
        
        // Let's use the approach: broadcast the reduced sum, then divide
        // Muls(reduceResult, reduceResult, 1.0f/dim, tileLength) then the first element = sum/dim = mean
        // Then Duplicate to broadcast mean
        // But we need to extract the scalar...
        
        // Better: use the sum tensor directly for division
        // Let's prepare a "mean" tensor by: take reduceResult, Muls by 1/dim, then Duplicate from [0]
        
        // Simplest correct approach: use Div element-wise with a broadcast mean tensor
        AscendC::Muls(reduceResult, reduceResult, 1.0f / (float)this->dim, 8);
        // Now reduceResult[0] = mean value of abs
        // Broadcast this to all elements
        // Use AscendC::Duplicate... but we need the scalar value
        // AscendC can't easily extract scalar from tensor
        
        // Alternative: use Brcb (broadcast) or just do Div with Adds trick
        // Let me use a different approach for the whole row processing
        
        // Second pass: divide x by mean using the sumLocal
        // We have sumLocal which is the accumulated abs values tile-wise
        // That's NOT what we want - we need the TOTAL sum
        
        // Let me restructure: after reduction, reduceResult contains the total sum in element [0]
        // (approximately, with some extra from padding, but padding was 0)
        
        // Use AscendC::BlockDataCopy or just reuse Brcb
        // Actually: AscendC has Brcb(dst, src, ...) for broadcast copy
        // Or we can do: Duplicate the first 8 elements across the whole tensor 
        
        // The simplest: copy element[0] block to all blocks
        // reduceResult[0] has sum/dim = mean
        // Use Brcb to broadcast
        AscendC::BrcbRepeat(scalarLocal, reduceResult, this->tileLength / 8, 1, 8);
        // Now scalarLocal is filled with the mean value
        
        // Second pass: divide each tile by mean
        remaining = this->dim;
        offset = 0;
        for (uint32_t t = 0; t < this->tileNum; t++) {
            if (remaining == 0) break;
            uint32_t curLen = this->tileLength;
            if (curLen > remaining) {
                curLen = remaining;
            }
            uint32_t copyLen = ((curLen + 7) / 8) * 8;
            if (copyLen > this->tileLength) copyLen = this->tileLength;

            AscendC::LocalTensor<float> inLocal = inQueue.AllocTensor<float>();
            AscendC::DataCopy(inLocal, xGm[rowOffset + offset], copyLen);
            inQueue.EnQue(inLocal);

            AscendC::LocalTensor<float> inLocal2 = inQueue.DeQue<float>();
            AscendC::LocalTensor<float> outLocal = outQueue.AllocTensor<float>();
            AscendC::Div(outLocal, inLocal2, scalarLocal, copyLen);
            outQueue.EnQue(outLocal);
            inQueue.FreeTensor(inLocal2);

            AscendC::LocalTensor<float> outLocal2 = outQueue.DeQue<float>();
            AscendC::DataCopy(yGm[rowOffset + offset], outLocal2, copyLen);
            outQueue.FreeTensor(outLocal2);

            offset += curLen;
            remaining -= curLen;
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> workBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scalarBuf;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t batchSize;
    uint32_t dim;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t rowsPerBlock;
    uint32_t startRow;
    uint32_t endRow;
};

extern "C" __global__ __aicore__ void l1_norm_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelL1Norm op;
    op.Init(x, y, tiling_data.batchSize, tiling_data.dim, tiling_data.tileNum);
    op.Process();
}
