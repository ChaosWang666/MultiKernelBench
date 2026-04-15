
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelCumprod {
public:
    __aicore__ inline KernelCumprod() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t dimSize, uint32_t outerSize, uint32_t innerSize)
    {
        this->totalLength = totalLength;
        this->dimSize = dimSize;
        this->outerSize = outerSize;
        this->innerSize = innerSize;

        // Total number of "lines" along the cumprod dimension
        // Each line is identified by (outerIdx, innerIdx) and has length dimSize
        this->totalLines = outerSize * innerSize;

        // Distribute lines across blocks
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t linesPerBlock = (this->totalLines + blockNum - 1) / blockNum;
        this->lineStart = blockIdx * linesPerBlock;
        this->lineEnd = (this->lineStart + linesPerBlock > this->totalLines) ? this->totalLines : this->lineStart + linesPerBlock;

        xGm.SetGlobalBuffer((__gm__ float *)x, totalLength);
        yGm.SetGlobalBuffer((__gm__ float *)y, totalLength);
    }
    __aicore__ inline void Process()
    {
        // For the case innerSize == 1 (dim is the last dimension), each line is contiguous
        // For innerSize > 1, elements are strided
        if (this->innerSize == 1) {
            ProcessContiguous();
        } else {
            ProcessStrided();
        }
    }

private:
    __aicore__ inline void ProcessContiguous()
    {
        // Each line is contiguous in memory with stride 1 between consecutive elements
        // line index l corresponds to outerIdx = l, innerIdx = 0
        // base offset = outerIdx * dimSize * innerSize = l * dimSize
        for (uint32_t l = this->lineStart; l < this->lineEnd; l++) {
            uint32_t baseOffset = l * this->dimSize;
            float acc = 1.0f;
            for (uint32_t d = 0; d < this->dimSize; d++) {
                float val;
                // Scalar read
                AscendC::DataCopy(tmpBuf, xGm[baseOffset + d], 32);  // min 32 bytes
                val = tmpBuf.GetValue(0);
                acc *= val;
                tmpBuf.SetValue(0, acc);
                AscendC::DataCopy(yGm[baseOffset + d], tmpBuf, 32);
            }
        }
    }

    __aicore__ inline void ProcessStrided()
    {
        for (uint32_t l = this->lineStart; l < this->lineEnd; l++) {
            uint32_t outerIdx = l / this->innerSize;
            uint32_t innerIdx = l % this->innerSize;
            uint32_t baseOffset = outerIdx * this->dimSize * this->innerSize + innerIdx;
            float acc = 1.0f;
            for (uint32_t d = 0; d < this->dimSize; d++) {
                uint32_t offset = baseOffset + d * this->innerSize;
                float val;
                AscendC::DataCopy(tmpBuf, xGm[offset], 32);
                val = tmpBuf.GetValue(0);
                acc *= val;
                tmpBuf.SetValue(0, acc);
                AscendC::DataCopy(yGm[offset], tmpBuf, 32);
            }
        }
    }

private:
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECIN> tmpBufManager;
    AscendC::LocalTensor<float> tmpBuf;
    uint32_t totalLength;
    uint32_t dimSize;
    uint32_t outerSize;
    uint32_t innerSize;
    uint32_t totalLines;
    uint32_t lineStart;
    uint32_t lineEnd;
};

// Alternative more efficient approach: use vectorized processing for contiguous case
class KernelCumprodOptimized {
public:
    __aicore__ inline KernelCumprodOptimized() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t dimSize, uint32_t outerSize, uint32_t innerSize)
    {
        this->totalLength = totalLength;
        this->dimSize = dimSize;
        this->outerSize = outerSize;
        this->innerSize = innerSize;
        this->totalLines = outerSize * innerSize;

        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();

        if (innerSize == 1) {
            // Contiguous lines: distribute lines across blocks
            uint32_t linesPerBlock = (this->totalLines + blockNum - 1) / blockNum;
            this->lineStart = blockIdx * linesPerBlock;
            this->lineEnd = (this->lineStart + linesPerBlock > this->totalLines) ? this->totalLines : this->lineStart + linesPerBlock;
        } else {
            // Strided: distribute outer dimension across blocks, process inner vectorially
            uint32_t outerPerBlock = (outerSize + blockNum - 1) / blockNum;
            this->outerStart = blockIdx * outerPerBlock;
            this->outerEnd = (this->outerStart + outerPerBlock > outerSize) ? outerSize : this->outerStart + outerPerBlock;
        }

        xGm.SetGlobalBuffer((__gm__ float *)x, totalLength);
        yGm.SetGlobalBuffer((__gm__ float *)y, totalLength);

        // Allocate buffer for vectorized inner processing
        // Align to 32 bytes (8 floats)
        uint32_t alignedInner = ((innerSize + 7) / 8) * 8;
        if (alignedInner < 8) alignedInner = 8;
        this->alignedInner = alignedInner;

        if (innerSize > 1) {
            pipe.InitBuffer(inQueue, BUFFER_NUM, alignedInner * sizeof(float));
            pipe.InitBuffer(outQueue, BUFFER_NUM, alignedInner * sizeof(float));
        } else {
            pipe.InitBuffer(inQueue, 1, 32);  // minimum 32 bytes
            pipe.InitBuffer(outQueue, 1, 32);
        }
    }

    __aicore__ inline void Process()
    {
        if (this->innerSize == 1) {
            ProcessContiguous();
        } else {
            ProcessVectorized();
        }
    }

private:
    __aicore__ inline void ProcessContiguous()
    {
        for (uint32_t l = this->lineStart; l < this->lineEnd; l++) {
            uint32_t baseOffset = l * this->dimSize;
            float acc = 1.0f;
            for (uint32_t d = 0; d < this->dimSize; d++) {
                AscendC::LocalTensor<float> inLocal = inQueue.AllocTensor<float>();
                AscendC::DataCopy(inLocal, xGm[baseOffset + d], 8); // copy min 32 bytes = 8 floats
                AscendC::SetAtomicNone();
                float val = inLocal.GetValue(0);
                acc *= val;
                inLocal.SetValue(0, acc);
                AscendC::DataCopy(yGm[baseOffset + d], inLocal, 8);
                inQueue.FreeTensor(inLocal);
            }
        }
    }

    __aicore__ inline void ProcessVectorized()
    {
        // For each outer index, process all inner elements vectorially
        for (uint32_t o = this->outerStart; o < this->outerEnd; o++) {
            uint32_t baseOffset = o * this->dimSize * this->innerSize;

            // First step: copy first slice and set as accumulator
            AscendC::LocalTensor<float> accLocal = outQueue.AllocTensor<float>();
            AscendC::DataCopy(accLocal, xGm[baseOffset], this->alignedInner);
            // Write first slice to output
            AscendC::DataCopy(yGm[baseOffset], accLocal, this->alignedInner);
            AscendC::SetAtomicNone();

            for (uint32_t d = 1; d < this->dimSize; d++) {
                uint32_t offset = baseOffset + d * this->innerSize;
                AscendC::LocalTensor<float> inLocal = inQueue.AllocTensor<float>();
                AscendC::DataCopy(inLocal, xGm[offset], this->alignedInner);
                // accLocal = accLocal * inLocal
                AscendC::Mul(accLocal, accLocal, inLocal, this->alignedInner);
                AscendC::DataCopy(yGm[offset], accLocal, this->alignedInner);
                inQueue.FreeTensor(inLocal);
            }
            outQueue.FreeTensor(accLocal);
        }
    }

private:
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueue;
    uint32_t totalLength;
    uint32_t dimSize;
    uint32_t outerSize;
    uint32_t innerSize;
    uint32_t totalLines;
    uint32_t lineStart;
    uint32_t lineEnd;
    uint32_t outerStart;
    uint32_t outerEnd;
    uint32_t alignedInner;
};

extern "C" __global__ __aicore__ void cumprod_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelCumprodOptimized op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.dimSize, tiling_data.outerSize, tiling_data.innerSize);
    op.Process();
}
