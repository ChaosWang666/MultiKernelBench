
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelMatmulSigmoidSum {
public:
    __aicore__ inline KernelMatmulSigmoidSum() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t batchSize, uint32_t hiddenSize,
                                 uint32_t tileLength, uint32_t numTiles, uint32_t rowsPerCore)
    {
        this->hiddenSize = hiddenSize;
        this->tileLength = tileLength;
        this->numTiles = numTiles;

        uint32_t blockId = AscendC::GetBlockIdx();
        uint32_t start = blockId * rowsPerCore;
        uint32_t end = start + rowsPerCore;
        if (end > batchSize) end = batchSize;
        this->startRow = start;
        this->numRows = (end > start) ? (end - start) : 0;

        xGm.SetGlobalBuffer((__gm__ float *)x, batchSize * hiddenSize);
        yGm.SetGlobalBuffer((__gm__ float *)y, batchSize);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(float));
        pipe.InitBuffer(workBuf, tileLength * sizeof(float));
        pipe.InitBuffer(reduceTmpBuf, tileLength * sizeof(float));
        pipe.InitBuffer(scalarBuf, 32);
        pipe.InitBuffer(accBuf, 32);
        pipe.InitBuffer(resultBuf, 64 * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (numRows == 0) return;

        AscendC::LocalTensor<float> resultLocal = resultBuf.Get<float>();
        AscendC::LocalTensor<float> accLocal = accBuf.Get<float>();
        AscendC::LocalTensor<float> scalar = scalarBuf.Get<float>();
        AscendC::LocalTensor<float> reduceTmp = reduceTmpBuf.Get<float>();
        AscendC::LocalTensor<float> work = workBuf.Get<float>();

        AscendC::Duplicate<float>(scalar, 0.0f, 8);

        for (uint32_t r = 0; r < numRows; r++) {
            uint32_t row = startRow + r;

            AscendC::Duplicate<float>(accLocal, 0.0f, 8);

            for (uint32_t t = 0; t < numTiles; t++) {
                AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
                AscendC::DataCopy(xLocal, xGm[row * hiddenSize + t * tileLength], tileLength);
                inQueueX.EnQue(xLocal);

                AscendC::LocalTensor<float> xIn = inQueueX.DeQue<float>();
                AscendC::Sigmoid<float>(work, xIn, tileLength);
                AscendC::ReduceSum<float, true>(scalar, work, reduceTmp, static_cast<int32_t>(tileLength));
                AscendC::Add<float>(accLocal, accLocal, scalar, 8);
                inQueueX.FreeTensor(xIn);
            }

            AscendC::SetFlag<AscendC::HardEvent::V_S>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(EVENT_ID0);

            float rowSum = accLocal.GetValue(0);
            resultLocal.SetValue(r, rowSum);
        }

        AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);

        AscendC::DataCopyExtParams copyOut;
        copyOut.blockCount = 1;
        copyOut.blockLen = numRows * sizeof(float);
        copyOut.srcStride = 0;
        copyOut.dstStride = 0;
        copyOut.rsv = 0;
        AscendC::DataCopyPad(yGm[startRow], resultLocal, copyOut);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TBuf<AscendC::TPosition::VECCALC> workBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceTmpBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scalarBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> accBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> resultBuf;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t hiddenSize;
    uint32_t tileLength;
    uint32_t numTiles;
    uint32_t startRow;
    uint32_t numRows;
};

extern "C" __global__ __aicore__ void matmul_sigmoid_sum_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelMatmulSigmoidSum op;
    op.Init(x, y, tiling_data.batchSize, tiling_data.hiddenSize, tiling_data.tileLength,
            tiling_data.numTiles, tiling_data.rowsPerCore);
    op.Process();
}
