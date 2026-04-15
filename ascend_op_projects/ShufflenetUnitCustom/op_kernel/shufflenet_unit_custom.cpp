
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelShufflenetUnit {
public:
    __aicore__ inline KernelShufflenetUnit() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR conv1_weight, GM_ADDR conv1_bias,
                                GM_ADDR bn1_scale, GM_ADDR bn1_offset,
                                GM_ADDR conv2_weight, GM_ADDR conv2_bias,
                                GM_ADDR bn2_scale, GM_ADDR bn2_offset,
                                GM_ADDR conv3_weight, GM_ADDR conv3_bias,
                                GM_ADDR bn3_scale, GM_ADDR bn3_offset,
                                GM_ADDR shortcut_conv_weight, GM_ADDR shortcut_conv_bias,
                                GM_ADDR shortcut_bn_scale, GM_ADDR shortcut_bn_offset,
                                GM_ADDR y,
                                uint32_t batchSize, uint32_t inChannels, uint32_t outChannels,
                                uint32_t height, uint32_t width, uint32_t groups, uint32_t midChannels)
    {
        this->batchSize = batchSize;
        this->inChannels = inChannels;
        this->outChannels = outChannels;
        this->height = height;
        this->width = width;
        this->groups = groups;
        this->midChannels = midChannels;
        this->totalElements = batchSize * inChannels * height * width;
        this->channelsPerGroup = inChannels / groups;
        this->tileSize = 1024;

        xGm.SetGlobalBuffer((__gm__ float *)x, this->totalElements);
        yGm.SetGlobalBuffer((__gm__ float *)y, this->totalElements);
        conv1WeightGm.SetGlobalBuffer((__gm__ float *)conv1_weight, inChannels * midChannels * 1 * 1);
        conv2WeightGm.SetGlobalBuffer((__gm__ float *)conv2_weight, midChannels * midChannels * 3 * 3);
        conv3WeightGm.SetGlobalBuffer((__gm__ float *)conv3_weight, midChannels * outChannels * 1 * 1);
        bn1ScaleGm.SetGlobalBuffer((__gm__ float *)bn1_scale, midChannels);
        bn1OffsetGm.SetGlobalBuffer((__gm__ float *)bn1_offset, midChannels);
        bn2ScaleGm.SetGlobalBuffer((__gm__ float *)bn2_scale, midChannels);
        bn2OffsetGm.SetGlobalBuffer((__gm__ float *)bn2_offset, midChannels);
        bn3ScaleGm.SetGlobalBuffer((__gm__ float *)bn3_scale, outChannels);
        bn3OffsetGm.SetGlobalBuffer((__gm__ float *)bn3_offset, outChannels);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileSize * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileSize * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        // Conv1 + BN1 + ReLU
        ConvBnRelu1();
        // Depthwise Conv2 + BN2
        DepthwiseConvBn2();
        // Shuffle
        Shuffle();
        // Conv3 + BN3 + ReLU
        ConvBnRelu3();
        // Shortcut
        Shortcut();
    }

private:
    __aicore__ inline void ConvBnRelu1()
    {
        // Placeholder for actual implementation
        // This would involve 1x1 group convolution, batch norm, and ReLU
    }
    __aicore__ inline void DepthwiseConvBn2()
    {
        // Placeholder for actual implementation
        // This would involve 3x3 depthwise convolution and batch norm
    }
    __aicore__ inline void Shuffle()
    {
        // Placeholder for actual implementation
        // This would involve channel shuffle operation
    }
    __aicore__ inline void ConvBnRelu3()
    {
        // Placeholder for actual implementation
        // This would involve 1x1 group convolution, batch norm, and ReLU
    }
    __aicore__ inline void Shortcut()
    {
        // Placeholder for actual implementation
        // This would involve shortcut connection
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    AscendC::GlobalTensor<float> conv1WeightGm;
    AscendC::GlobalTensor<float> conv2WeightGm;
    AscendC::GlobalTensor<float> conv3WeightGm;
    AscendC::GlobalTensor<float> bn1ScaleGm;
    AscendC::GlobalTensor<float> bn1OffsetGm;
    AscendC::GlobalTensor<float> bn2ScaleGm;
    AscendC::GlobalTensor<float> bn2OffsetGm;
    AscendC::GlobalTensor<float> bn3ScaleGm;
    AscendC::GlobalTensor<float> bn3OffsetGm;
    uint32_t batchSize;
    uint32_t inChannels;
    uint32_t outChannels;
    uint32_t height;
    uint32_t width;
    uint32_t groups;
    uint32_t midChannels;
    uint32_t totalElements;
    uint32_t channelsPerGroup;
    uint32_t tileSize;
};

extern "C" __global__ __aicore__ void shufflenet_unit_custom(
    GM_ADDR x, GM_ADDR conv1_weight, GM_ADDR conv1_bias,
    GM_ADDR bn1_scale, GM_ADDR bn1_offset,
    GM_ADDR conv2_weight, GM_ADDR conv2_bias,
    GM_ADDR bn2_scale, GM_ADDR bn2_offset,
    GM_ADDR conv3_weight, GM_ADDR conv3_bias,
    GM_ADDR bn3_scale, GM_ADDR bn3_offset,
    GM_ADDR shortcut_conv_weight, GM_ADDR shortcut_conv_bias,
    GM_ADDR shortcut_bn_scale, GM_ADDR shortcut_bn_offset,
    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelShufflenetUnit op;
    op.Init(x, conv1_weight, conv1_bias, bn1_scale, bn1_offset,
            conv2_weight, conv2_bias, bn2_scale, bn2_offset,
            conv3_weight, conv3_bias, bn3_scale, bn3_offset,
            shortcut_conv_weight, shortcut_conv_bias, shortcut_bn_scale, shortcut_bn_offset,
            y, tiling_data.batchSize, tiling_data.inChannels, tiling_data.outChannels,
            tiling_data.height, tiling_data.width, tiling_data.groups, tiling_data.midChannels);
    op.Process();
}
