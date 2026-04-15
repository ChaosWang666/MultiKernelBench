
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelResnetBasicBlock {
public:
    __aicore__ inline KernelResnetBasicBlock() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR conv1Weight, GM_ADDR conv1Bias,
                                GM_ADDR bn1Scale, GM_ADDR bn1Bias,
                                GM_ADDR conv2Weight, GM_ADDR conv2Bias,
                                GM_ADDR bn2Scale, GM_ADDR bn2Bias,
                                GM_ADDR downsampleConvWeight, GM_ADDR downsampleConvBias,
                                GM_ADDR downsampleBnScale, GM_ADDR downsampleBnBias,
                                GM_ADDR out, uint32_t batch, uint32_t inChannels,
                                uint32_t outChannels, uint32_t height, uint32_t width,
                                uint32_t stride, uint32_t useDownsample)
    {
        this->batch = batch;
        this->inChannels = inChannels;
        this->outChannels = outChannels;
        this->height = height;
        this->width = width;
        this->stride = stride;
        this->useDownsample = useDownsample;

        this->blockLength = batch * inChannels * height * width / AscendC::GetBlockNum();
        this->tileLength = this->blockLength / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ float *)x, this->blockLength);
        outGm.SetGlobalBuffer((__gm__ float *)out, this->blockLength);
        conv1WeightGm.SetGlobalBuffer((__gm__ float *)conv1Weight, 3 * 3 * inChannels * outChannels);
        conv1BiasGm.SetGlobalBuffer((__gm__ float *)conv1Bias, outChannels);
        bn1ScaleGm.SetGlobalBuffer((__gm__ float *)bn1Scale, outChannels);
        bn1BiasGm.SetGlobalBuffer((__gm__ float *)bn1Bias, outChannels);
        conv2WeightGm.SetGlobalBuffer((__gm__ float *)conv2Weight, 3 * 3 * outChannels * outChannels);
        conv2BiasGm.SetGlobalBuffer((__gm__ float *)conv2Bias, outChannels);
        bn2ScaleGm.SetGlobalBuffer((__gm__ float *)bn2Scale, outChannels);
        bn2BiasGm.SetGlobalBuffer((__gm__ float *)bn2Bias, outChannels);
        if (useDownsample) {
            downsampleConvWeightGm.SetGlobalBuffer((__gm__ float *)downsampleConvWeight, 1 * 1 * inChannels * outChannels);
            downsampleConvBiasGm.SetGlobalBuffer((__gm__ float *)downsampleConvBias, outChannels);
            downsampleBnScaleGm.SetGlobalBuffer((__gm__ float *)downsampleBnScale, outChannels);
            downsampleBnBiasGm.SetGlobalBuffer((__gm__ float *)downsampleBnBias, outChannels);
        }

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueConv1, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueBn1, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueConv2, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueBn2, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueIdentity, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueAdd, BUFFER_NUM, this->tileLength * sizeof(float));
        pipe.InitBuffer(outQueueRelu, BUFFER_NUM, this->tileLength * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t batchIdx = 0; batchIdx < batch; ++batchIdx) {
            ProcessBatch(batchIdx);
        }
    }

private:
    __aicore__ inline void ProcessBatch(uint32_t batchIdx)
    {
        uint32_t offset = batchIdx * inChannels * height * width;
        xGm.SetGlobalBuffer((__gm__ float *)xGm.GetBaseAddr() + offset, inChannels * height * width);
        outGm.SetGlobalBuffer((__gm__ float *)outGm.GetBaseAddr() + offset, inChannels * height * width);

        // Conv1 + BN1 + ReLU
        ConvBnRelu1(offset);

        // Conv2 + BN2
        ConvBn2(offset);

        // Identity path
        Identity(offset);

        // Add + ReLU
        AddRelu(offset);
    }

    __aicore__ inline void ConvBnRelu1(uint32_t offset)
    {
        // Conv1
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::DataCopy(xLocal, xGm, inChannels * height * width);
        inQueueX.EnQue(xLocal);

        AscendC::LocalTensor<float> conv1Out = outQueueConv1.AllocTensor<float>();
        AscendC::Conv2d(conv1Out, xLocal, conv1WeightGm, conv1BiasGm, 3, 3, 1, 1, 1, 1, 1, 1);
        outQueueConv1.EnQue(conv1Out);

        // BN1
        AscendC::LocalTensor<float> bn1Out = outQueueBn1.AllocTensor<float>();
        AscendC::BatchNorm2d(bn1Out, conv1Out, bn1ScaleGm, bn1BiasGm, 1e-5f);
        outQueueBn1.EnQue(bn1Out);

        // ReLU
        AscendC::LocalTensor<float> reluOut = outQueueRelu.AllocTensor<float>();
        AscendC::ReLU(reluOut, bn1Out);
        outQueueRelu.EnQue(reluOut);
    }

    __aicore__ inline void ConvBn2(uint32_t offset)
    {
        // Conv2
        AscendC::LocalTensor<float> xLocal = outQueueRelu.DeQue<float>();
        AscendC::LocalTensor<float> conv2Out = outQueueConv2.AllocTensor<float>();
        AscendC::Conv2d(conv2Out, xLocal, conv2WeightGm, conv2BiasGm, 3, 3, 1, 1, 1, 1, 1, 1);
        outQueueConv2.EnQue(conv2Out);

        // BN2
        AscendC::LocalTensor<float> bn2Out = outQueueBn2.AllocTensor<float>();
        AscendC::BatchNorm2d(bn2Out, conv2Out, bn2ScaleGm, bn2BiasGm, 1e-5f);
        outQueueBn2.EnQue(bn2Out);
    }

    __aicore__ inline void Identity(uint32_t offset)
    {
        // Identity path
        AscendC::LocalTensor<float> identityOut = outQueueIdentity.AllocTensor<float>();
        if (useDownsample) {
            // Downsample
            AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
            AscendC::LocalTensor<float> downsampleOut = outQueueIdentity.AllocTensor<float>();
            AscendC::Conv2d(downsampleOut, xLocal, downsampleConvWeightGm, downsampleConvBiasGm, 1, 1, 0, 0, 1, 1, 1, 1);
            outQueueIdentity.EnQue(downsampleOut);
        } else {
            // Direct copy
            AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
            AscendC::DataCopy(identityOut, xLocal, inChannels * height * width);
            outQueueIdentity.EnQue(identityOut);
        }
    }

    __aicore__ inline void AddRelu(uint32_t offset)
    {
        // Add
        AscendC::LocalTensor<float> xLocal = outQueueBn2.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueueIdentity.DeQue<float>();
        AscendC::LocalTensor<float> addOut = outQueueAdd.AllocTensor<float>();
        AscendC::Add(addOut, xLocal, yLocal, inChannels * height * width);
        outQueueAdd.EnQue(addOut);

        // ReLU
        AscendC::LocalTensor<float> reluOut = outQueueRelu.AllocTensor<float>();
        AscendC::ReLU(reluOut, addOut);
        outQueueRelu.EnQue(reluOut);

        // Copy back to global memory
        AscendC::LocalTensor<float> finalOut = outQueueRelu.DeQue<float>();
        AscendC::DataCopy(outGm, finalOut, inChannels * height * width);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueConv1, outQueueBn1, outQueueConv2, outQueueBn2, outQueueIdentity, outQueueAdd, outQueueRelu;
    AscendC::GlobalTensor<float> xGm, outGm;
    AscendC::GlobalTensor<float> conv1WeightGm, conv1BiasGm;
    AscendC::GlobalTensor<float> bn1ScaleGm, bn1BiasGm;
    AscendC::GlobalTensor<float> conv2WeightGm, conv2BiasGm;
    AscendC::GlobalTensor<float> bn2ScaleGm, bn2BiasGm;
    AscendC::GlobalTensor<float> downsampleConvWeightGm, downsampleConvBiasGm;
    AscendC::GlobalTensor<float> downsampleBnScaleGm, downsampleBnBiasGm;
    uint32_t batch;
    uint32_t inChannels;
    uint32_t outChannels;
    uint32_t height;
    uint32_t width;
    uint32_t stride;
    uint32_t useDownsample;
    uint32_t blockLength;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void resnet_basic_block_custom(
    GM_ADDR x, GM_ADDR conv1_weight, GM_ADDR conv1_bias,
    GM_ADDR bn1_scale, GM_ADDR bn1_bias,
    GM_ADDR conv2_weight, GM_ADDR conv2_bias,
    GM_ADDR bn2_scale, GM_ADDR bn2_bias,
    GM_ADDR downsample_conv_weight, GM_ADDR downsample_conv_bias,
    GM_ADDR downsample_bn_scale, GM_ADDR downsample_bn_bias,
    GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelResnetBasicBlock op;
    op.Init(x, conv1_weight, conv1_bias, bn1_scale, bn1_bias,
            conv2_weight, conv2_bias, bn2_scale, bn2_bias,
            downsample_conv_weight, downsample_conv_bias,
            downsample_bn_scale, downsample_bn_bias,
            out, tiling_data.batch, tiling_data.inChannels,
            tiling_data.outChannels, tiling_data.height, tiling_data.width,
            tiling_data.stride, tiling_data.useDownsample);
    op.Process();
}
