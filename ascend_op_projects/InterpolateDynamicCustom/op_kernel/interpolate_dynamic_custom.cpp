
#include "kernel_operator.h"

class KernelInterpolateBilinear {
public:
    __aicore__ inline KernelInterpolateBilinear() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                 uint32_t batchSize, uint32_t channels,
                                 uint32_t inputH, uint32_t inputW,
                                 uint32_t outputH, uint32_t outputW,
                                 uint32_t totalOutputRows, uint32_t rowsPerCore,
                                 uint32_t remainRows)
    {
        this->batchSize = batchSize;
        this->channels = channels;
        this->inputH = inputH;
        this->inputW = inputW;
        this->outputH = outputH;
        this->outputW = outputW;
        this->inputHW = inputH * inputW;
        this->outputHW = outputH * outputW;

        uint32_t blockIdx = AscendC::GetBlockIdx();
        this->myRowStart = blockIdx * rowsPerCore + (blockIdx < remainRows ? blockIdx : remainRows);
        this->myRowCount = rowsPerCore + (blockIdx < remainRows ? 1 : 0);

        uint32_t totalInput = batchSize * channels * inputH * inputW;
        uint32_t totalOutput = batchSize * channels * outputH * outputW;
        xGm.SetGlobalBuffer((__gm__ float *)x, totalInput);
        yGm.SetGlobalBuffer((__gm__ float *)y, totalOutput);

        // Compute scale factors: (input - 1) / (output - 1) for align_corners=True
        // For align_corners=False: scale = input / output
        this->scaleH = (float)inputH / (float)outputH;
        this->scaleW = (float)inputW / (float)outputW;

        // Allocate buffers for processing one output row at a time
        // We need to read up to 2 input rows and produce 1 output row
        uint32_t alignedOutputW = ((outputW + 7) / 8) * 8;
        uint32_t alignedInputW = ((inputW + 7) / 8) * 8;
        this->alignedOutputW = alignedOutputW;
        this->alignedInputW = alignedInputW;

        pipe.InitBuffer(inQueueRow0, 1, alignedInputW * sizeof(float));
        pipe.InitBuffer(inQueueRow1, 1, alignedInputW * sizeof(float));
        pipe.InitBuffer(outQueue, 1, alignedOutputW * sizeof(float));
        pipe.InitBuffer(tmpBuf1, 1, alignedOutputW * sizeof(float));
        pipe.InitBuffer(tmpBuf2, 1, alignedOutputW * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < this->myRowCount; i++) {
            uint32_t globalRow = this->myRowStart + i;
            ProcessRow(globalRow);
        }
    }

private:
    __aicore__ inline void ProcessRow(uint32_t globalRow)
    {
        // globalRow indexes into (batch * channels * outputH)
        uint32_t bc = globalRow / outputH;
        uint32_t oh = globalRow % outputH;

        // Compute source y coordinate (align_corners=False)
        float srcY = ((float)oh + 0.5f) * scaleH - 0.5f;
        if (srcY < 0.0f) srcY = 0.0f;
        
        uint32_t iy0 = (uint32_t)srcY;
        uint32_t iy1 = iy0 + 1;
        if (iy0 >= inputH - 1) {
            iy0 = inputH - 1;
            iy1 = inputH - 1;
        }
        float fy = srcY - (float)iy0;

        // Input offset for this batch*channel
        uint32_t inputBaseOffset = bc * inputHW;
        uint32_t outputBaseOffset = bc * outputHW;

        // Read two input rows
        AscendC::LocalTensor<float> row0Local = inQueueRow0.AllocTensor<float>();
        AscendC::LocalTensor<float> row1Local = inQueueRow1.AllocTensor<float>();

        // Copy input rows - need to use aligned copy
        uint32_t copyLen = ((inputW + 7) / 8) * 8;
        AscendC::DataCopy(row0Local, xGm[inputBaseOffset + iy0 * inputW], copyLen);
        AscendC::DataCopy(row1Local, xGm[inputBaseOffset + iy1 * inputW], copyLen);

        AscendC::LocalTensor<float> outLocal = outQueue.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp1 = tmpBuf1.AllocTensor<float>();
        AscendC::LocalTensor<float> tmp2 = tmpBuf2.AllocTensor<float>();

        // For each output pixel, compute bilinear interpolation
        // We do this element-by-element since x coordinates differ per pixel
        for (uint32_t ow = 0; ow < outputW; ow++) {
            float srcX = ((float)ow + 0.5f) * scaleW - 0.5f;
            if (srcX < 0.0f) srcX = 0.0f;

            uint32_t ix0 = (uint32_t)srcX;
            uint32_t ix1 = ix0 + 1;
            if (ix0 >= inputW - 1) {
                ix0 = inputW - 1;
                ix1 = inputW - 1;
            }
            float fx = srcX - (float)ix0;

            float v00 = row0Local.GetValue(ix0);
            float v01 = row0Local.GetValue(ix1);
            float v10 = row1Local.GetValue(ix0);
            float v11 = row1Local.GetValue(ix1);

            float val = (1.0f - fy) * ((1.0f - fx) * v00 + fx * v01) +
                        fy * ((1.0f - fx) * v10 + fx * v11);

            outLocal.SetValue(ow, val);
        }

        // Write output row
        uint32_t outCopyLen = ((outputW + 7) / 8) * 8;
        AscendC::DataCopy(yGm[outputBaseOffset + oh * outputW], outLocal, outCopyLen);

        tmpBuf2.FreeTensor(tmp2);
        tmpBuf1.FreeTensor(tmp1);
        outQueue.FreeTensor(outLocal);
        inQueueRow1.FreeTensor(row1Local);
        inQueueRow0.FreeTensor(row0Local);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueRow0, inQueueRow1;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> tmpBuf1, tmpBuf2;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    uint32_t batchSize, channels, inputH, inputW, outputH, outputW;
    uint32_t inputHW, outputHW;
    uint32_t myRowStart, myRowCount;
    uint32_t alignedOutputW, alignedInputW;
    float scaleH, scaleW;
};

extern "C" __global__ __aicore__ void interpolate_dynamic_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelInterpolateBilinear op;
    op.Init(x, y,
            tiling_data.batchSize, tiling_data.channels,
            tiling_data.inputH, tiling_data.inputW,
            tiling_data.outputH, tiling_data.outputW,
            tiling_data.totalOutputRows, tiling_data.rowsPerCore,
            tiling_data.remainRows);
    op.Process();
}
