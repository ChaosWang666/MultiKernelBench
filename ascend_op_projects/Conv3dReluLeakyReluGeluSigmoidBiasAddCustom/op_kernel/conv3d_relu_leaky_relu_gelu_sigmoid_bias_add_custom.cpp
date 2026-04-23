
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

class KernelConvActBias {
public:
    __aicore__ inline KernelConvActBias() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR bias, GM_ADDR y,
        uint32_t totalTasks, uint32_t channels, uint32_t elemsPerChannel, uint32_t tileLen)
    {
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();

        uint32_t tasksPerBlock = totalTasks / blockNum;
        uint32_t remainder = totalTasks % blockNum;
        if (blockIdx < remainder) {
            myTasks = tasksPerBlock + 1;
            startTask = blockIdx * myTasks;
        } else {
            myTasks = tasksPerBlock;
            startTask = remainder * (tasksPerBlock + 1) + (blockIdx - remainder) * tasksPerBlock;
        }

        this->channels = channels;
        this->elemsPerChannel = elemsPerChannel;
        this->tileLen = tileLen;

        xGm.SetGlobalBuffer((__gm__ float *)x);
        biasGm.SetGlobalBuffer((__gm__ float *)bias, channels);
        yGm.SetGlobalBuffer((__gm__ float *)y);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLen * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, tileLen * sizeof(float));

        uint32_t biasAlignedSize = ((channels * sizeof(float) + 31) / 32) * 32;
        if (biasAlignedSize < 32) { biasAlignedSize = 32; }
        pipe.InitBuffer(biasQueue, 1, biasAlignedSize);

        pipe.InitBuffer(workBuf, tileLen * sizeof(float));

        AscendC::LocalTensor<float> biasLocal = biasQueue.AllocTensor<float>();
        AscendC::DataCopyExtParams biasCopyParams;
        biasCopyParams.blockCount = 1;
        biasCopyParams.blockLen = channels * sizeof(float);
        biasCopyParams.srcStride = 0;
        biasCopyParams.dstStride = 0;
        AscendC::DataCopyPadExtParams<float> biasPadParams;
        biasPadParams.isPad = false;
        biasPadParams.leftPadding = 0;
        biasPadParams.rightPadding = 0;
        biasPadParams.paddingValue = 0.0f;
        AscendC::DataCopyPad(biasLocal, biasGm, biasCopyParams, biasPadParams);
        biasQueue.EnQue(biasLocal);
    }

    __aicore__ inline void Process()
    {
        if (myTasks == 0) {
            AscendC::LocalTensor<float> biasLocal = biasQueue.DeQue<float>();
            biasQueue.FreeTensor(biasLocal);
            return;
        }

        AscendC::LocalTensor<float> biasLocal = biasQueue.DeQue<float>();

        for (uint32_t t = 0; t < myTasks; t++) {
            uint32_t taskIdx = startTask + t;
            uint32_t c = taskIdx % channels;
            float biasVal = biasLocal.GetValue(c);

            uint64_t gmOffset = (uint64_t)taskIdx * (uint64_t)elemsPerChannel;
            uint32_t numTiles = (elemsPerChannel + tileLen - 1) / tileLen;

            for (uint32_t tile = 0; tile < numTiles; tile++) {
                uint32_t offset = tile * tileLen;
                uint32_t curLen = (offset + tileLen <= elemsPerChannel) ? tileLen : (elemsPerChannel - offset);
                ProcessTile(gmOffset + (uint64_t)offset, curLen, biasVal);
            }
        }

        biasQueue.FreeTensor(biasLocal);
    }

private:
    __aicore__ inline void ProcessTile(uint64_t gmOff, uint32_t len, float biasVal)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = len * sizeof(float);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        AscendC::DataCopyPadExtParams<float> padParams;
        padParams.isPad = false;
        padParams.leftPadding = 0;
        padParams.rightPadding = 0;
        padParams.paddingValue = 0.0f;
        AscendC::DataCopyPad(xLocal, xGm[gmOff], copyInParams, padParams);
        inQueueX.EnQue(xLocal);

        AscendC::LocalTensor<float> xIn = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        AscendC::LocalTensor<float> work = workBuf.Get<float>();

        // Step 1: ReLU
        AscendC::Relu<float>(yLocal, xIn, len);

        // Step 2: LeakyReLU (identity for non-negative values, skipped)

        // Step 3: GELU (tanh approximation)
        // work = x^2
        AscendC::Mul<float>(work, yLocal, yLocal, len);
        // work = x^3
        AscendC::Mul<float>(work, work, yLocal, len);
        // work = 0.044715 * x^3
        AscendC::Muls<float>(work, work, 0.044715f, len);
        // work = x + 0.044715 * x^3
        AscendC::Add<float>(work, work, yLocal, len);
        // work = sqrt(2/pi) * work
        AscendC::Muls<float>(work, work, 0.7978845608028654f, len);
        // work = tanh(work)
        AscendC::Tanh<float>(work, work, len);
        // work = 1 + tanh(...)
        AscendC::Adds<float>(work, work, 1.0f, len);
        // yLocal = x * (1 + tanh(...))
        AscendC::Mul<float>(yLocal, yLocal, work, len);
        // yLocal = 0.5 * x * (1 + tanh(...))
        AscendC::Muls<float>(yLocal, yLocal, 0.5f, len);

        // Step 4: Sigmoid
        AscendC::Sigmoid<float>(yLocal, yLocal, len);

        // Step 5: Bias add
        AscendC::Adds<float>(yLocal, yLocal, biasVal, len);

        outQueueY.EnQue<float>(yLocal);
        inQueueX.FreeTensor(xIn);

        AscendC::LocalTensor<float> yOut = outQueueY.DeQue<float>();
        AscendC::DataCopyExtParams copyOutParams;
        copyOutParams.blockCount = 1;
        copyOutParams.blockLen = len * sizeof(float);
        copyOutParams.srcStride = 0;
        copyOutParams.dstStride = 0;
        AscendC::DataCopyPad(yGm[gmOff], yOut, copyOutParams);
        outQueueY.FreeTensor(yOut);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> biasQueue;
    AscendC::TBuf<AscendC::TPosition::VECCALC> workBuf;
    AscendC::GlobalTensor<float> xGm, biasGm, yGm;
    uint32_t startTask;
    uint32_t myTasks;
    uint32_t channels;
    uint32_t elemsPerChannel;
    uint32_t tileLen;
};

extern "C" __global__ __aicore__ void conv3d_relu_leaky_relu_gelu_sigmoid_bias_add_custom(
    GM_ADDR x, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelConvActBias op;
    op.Init(x, bias, y, tiling_data.totalTasks, tiling_data.channels,
            tiling_data.elemsPerChannel, tiling_data.tileLen);
    op.Process();
}
