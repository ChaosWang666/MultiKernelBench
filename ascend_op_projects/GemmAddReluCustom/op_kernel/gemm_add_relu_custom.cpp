
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;
using namespace matmul;

constexpr uint32_t RELU_TILE_LEN = 8192;

class KernelGemmAddRelu {
public:
    __aicore__ inline KernelGemmAddRelu() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR y,
                                 GM_ADDR workspace, const TCubeTiling& mmTiling) {
        this->tiling = mmTiling;

        xGm.SetGlobalBuffer((__gm__ float*)x, mmTiling.M * mmTiling.Ka);
        wGm.SetGlobalBuffer((__gm__ float*)weight, mmTiling.N * mmTiling.Kb);
        biasGm.SetGlobalBuffer((__gm__ float*)bias, mmTiling.N);
        yGm.SetGlobalBuffer((__gm__ float*)y, mmTiling.M * mmTiling.N);

        pipe.InitBuffer(inQueue, 2, RELU_TILE_LEN * sizeof(float));
        pipe.InitBuffer(outQueue, 2, RELU_TILE_LEN * sizeof(float));
    }

    __aicore__ inline void Process() {
        RunMatmul();
        SyncAll();
        RunRelu();
    }

    TPipe pipe;
    Matmul<MatmulType<TPosition::GM, CubeFormat::ND, float, false>,
           MatmulType<TPosition::GM, CubeFormat::ND, float, true>,
           MatmulType<TPosition::GM, CubeFormat::ND, float, false>,
           MatmulType<TPosition::GM, CubeFormat::ND, float, false>> matmulObj;

private:
    __aicore__ inline void RunMatmul() {
        int32_t blockIdx = GetBlockIdx();
        int32_t mCnt = (tiling.M + tiling.singleCoreM - 1) / tiling.singleCoreM;
        int32_t nCnt = (tiling.N + tiling.singleCoreN - 1) / tiling.singleCoreN;
        int32_t totalBlocks = mCnt * nCnt;

        if (blockIdx >= totalBlocks) {
            return;
        }

        int32_t mIdx = blockIdx / nCnt;
        int32_t nIdx = blockIdx % nCnt;

        int32_t offsetA = mIdx * tiling.singleCoreM * tiling.Ka;
        int32_t offsetB = nIdx * tiling.singleCoreN * tiling.Kb;
        int32_t offsetC = mIdx * tiling.singleCoreM * tiling.N + nIdx * tiling.singleCoreN;
        int32_t offsetBias = nIdx * tiling.singleCoreN;

        matmulObj.SetTensorA(xGm[offsetA]);
        matmulObj.SetTensorB(wGm[offsetB]);
        matmulObj.SetBias(biasGm[offsetBias]);
        matmulObj.IterateAll(yGm[offsetC]);
        matmulObj.End();
    }

    __aicore__ inline void RunRelu() {
        uint32_t blockIdx = GetBlockIdx();
        uint32_t numBlocks = GetBlockNum();
        uint32_t totalSize = tiling.M * tiling.N;

        uint32_t baseChunk = totalSize / numBlocks;
        uint32_t remainder = totalSize % numBlocks;

        uint32_t myChunk;
        uint32_t myStart;
        if (blockIdx < remainder) {
            myChunk = baseChunk + 1;
            myStart = blockIdx * myChunk;
        } else {
            myChunk = baseChunk;
            myStart = remainder * (baseChunk + 1) + (blockIdx - remainder) * baseChunk;
        }

        if (myChunk == 0) {
            return;
        }

        uint32_t processed = 0;
        while (processed < myChunk) {
            uint32_t remain = myChunk - processed;
            uint32_t thisTile = (remain < RELU_TILE_LEN) ? remain : RELU_TILE_LEN;
            uint32_t curOffset = myStart + processed;

            LocalTensor<float> xLocal = inQueue.AllocTensor<float>();
            DataCopy(xLocal, yGm[curOffset], thisTile);
            inQueue.EnQue(xLocal);

            LocalTensor<float> xIn = inQueue.DeQue<float>();
            LocalTensor<float> yLocal = outQueue.AllocTensor<float>();
            Relu(yLocal, xIn, thisTile);
            outQueue.EnQue(yLocal);
            inQueue.FreeTensor(xIn);

            LocalTensor<float> yOut = outQueue.DeQue<float>();
            DataCopy(yGm[curOffset], yOut, thisTile);
            outQueue.FreeTensor(yOut);

            processed += thisTile;
        }
    }

    GlobalTensor<float> xGm;
    GlobalTensor<float> wGm;
    GlobalTensor<float> biasGm;
    GlobalTensor<float> yGm;
    TCubeTiling tiling;
    TQue<TPosition::VECIN, 2> inQueue;
    TQue<TPosition::VECOUT, 2> outQueue;
};

extern "C" __global__ __aicore__ void gemm_add_relu_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling) {

    GET_TILING_DATA(tiling_data, tiling);

    KernelGemmAddRelu op;
    REGIST_MATMUL_OBJ(&op.pipe, GetSysWorkSpacePtr(), op.matmulObj, &tiling_data.cubeTilingData);
    op.Init(x, weight, bias, y, workspace, tiling_data.cubeTilingData);
    op.Process();
}
