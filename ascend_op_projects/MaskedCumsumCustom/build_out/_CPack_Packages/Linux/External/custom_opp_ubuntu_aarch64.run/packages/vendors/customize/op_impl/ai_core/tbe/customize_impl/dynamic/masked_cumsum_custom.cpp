
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;
// Maximum tile size for processing rows in chunks
constexpr int32_t TILE_SIZE = 512;

class KernelMaskedCumsum {
public:
    __aicore__ inline KernelMaskedCumsum() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR mask, GM_ADDR z,
                                 uint32_t totalRows, uint32_t rowLength)
    {
        this->totalRows = totalRows;
        this->rowLength = rowLength;

        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();

        // Distribute rows across blocks
        this->rowsPerBlock = (totalRows + blockNum - 1) / blockNum;
        this->startRow = blockIdx * this->rowsPerBlock;
        this->endRow = startRow + rowsPerBlock;
        if (this->endRow > totalRows) this->endRow = totalRows;

        // Set global buffers for the entire tensor
        xGm.SetGlobalBuffer((__gm__ float *)x, totalRows * rowLength);
        maskGm.SetGlobalBuffer((__gm__ uint8_t *)mask, totalRows * rowLength);
        zGm.SetGlobalBuffer((__gm__ float *)z, totalRows * rowLength);

        // Calculate tile parameters
        // We process each row in tiles of TILE_SIZE elements
        this->tileSize = TILE_SIZE;
        if (this->tileSize > rowLength) {
            // Round up to 32 bytes / 8 floats alignment
            this->tileSize = ((rowLength + 7) / 8) * 8;
        }

        // Allocate buffers
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileSize * sizeof(float));
        pipe.InitBuffer(inQueueMask, BUFFER_NUM, this->tileSize * sizeof(uint8_t));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileSize * sizeof(float));
        // Temp buffer for cast
        pipe.InitBuffer(tmpBuf, 1, this->tileSize * sizeof(float));
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
        uint32_t rowOffset = row * this->rowLength;
        float runningSum = 0.0f;

        uint32_t processed = 0;
        while (processed < this->rowLength) {
            uint32_t remaining = this->rowLength - processed;
            uint32_t curTile = this->tileSize;
            if (remaining < curTile) {
                curTile = ((remaining + 7) / 8) * 8; // align to 8 floats (32 bytes)
            }
            uint32_t validCount = remaining < this->tileSize ? remaining : this->tileSize;

            // Copy in x tile
            AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
            AscendC::DataCopy(xLocal, xGm[rowOffset + processed], curTile);
            inQueueX.EnQue(xLocal);

            // Copy in mask tile - mask is bool (uint8_t)
            // Need to align mask copy to 32 bytes
            uint32_t maskCopyLen = ((curTile + 31) / 32) * 32;
            AscendC::LocalTensor<uint8_t> maskLocal = inQueueMask.AllocTensor<uint8_t>();
            AscendC::DataCopy(maskLocal, maskGm[rowOffset + processed], maskCopyLen);
            inQueueMask.EnQue(maskLocal);

            // Compute
            xLocal = inQueueX.DeQue<float>();
            maskLocal = inQueueMask.DeQue<uint8_t>();
            AscendC::LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
            AscendC::LocalTensor<float> tmpLocal = tmpBuf.Get<float>();

            // Cast mask (uint8_t) to float: use Cast
            // First cast int8 to half then half to float, or use Duplicate + Select
            // Actually, let's do element-wise multiply: cast mask to float then multiply
            AscendC::Cast(tmpLocal, maskLocal, AscendC::RoundMode::CAST_NONE, curTile);
            AscendC::Mul(zLocal, xLocal, tmpLocal, curTile);

            // Now zLocal has x * mask, need cumsum
            // Cumsum is sequential - we need to do it element by element
            // Copy to do sequential scan
            for (uint32_t i = 0; i < validCount; i++) {
                float val = zLocal.GetValue(i);
                runningSum += val;
                zLocal.SetValue(i, runningSum);
            }

            outQueueZ.EnQue(zLocal);
            inQueueX.FreeTensor(xLocal);
            inQueueMask.FreeTensor(maskLocal);

            // Copy out
            AscendC::LocalTensor<float> zOut = outQueueZ.DeQue<float>();
            AscendC::DataCopy(zGm[rowOffset + processed], zOut, curTile);
            outQueueZ.FreeTensor(zOut);

            processed += validCount;
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueMask;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<uint8_t> maskGm;
    AscendC::GlobalTensor<float> zGm;
    uint32_t totalRows;
    uint32_t rowLength;
    uint32_t rowsPerBlock;
    uint32_t startRow;
    uint32_t endRow;
    uint32_t tileSize;
};

extern "C" __global__ __aicore__ void masked_cumsum_custom(GM_ADDR x, GM_ADDR mask, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelMaskedCumsum op;
    op.Init(x, mask, z, tiling_data.totalRows, tiling_data.rowLength);
    op.Process();
}
