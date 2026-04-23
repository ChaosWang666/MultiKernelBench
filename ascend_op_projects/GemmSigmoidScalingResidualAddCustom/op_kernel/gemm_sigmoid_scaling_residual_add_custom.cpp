
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;
using namespace matmul;

class KernelGemmSigmoidScalingResidualAdd {
public:
    __aicore__ inline KernelGemmSigmoidScalingResidualAdd() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR y,
                                 uint32_t M, uint32_t N, uint32_t K, float scalingFactor) {
        this->M = M;
        this->N = N;
        this->K = K;
        this->scalingFactor = scalingFactor;
        xGm.SetGlobalBuffer((__gm__ float*)x, (uint64_t)M * K);
        weightGm.SetGlobalBuffer((__gm__ float*)weight, (uint64_t)N * K);
        biasGm.SetGlobalBuffer((__gm__ float*)bias, N);
        yGm.SetGlobalBuffer((__gm__ float*)y, (uint64_t)M * N);
    }

    __aicore__ inline void ProcessMatmul() {
        mm.SetTensorA(xGm);
        mm.SetTensorB(weightGm, true);
        mm.SetBias(biasGm);
        mm.IterateAll(yGm);
        mm.End();
    }

    __aicore__ inline void ProcessVector() {
        constexpr uint32_t TILE = 4096;
        pipe.InitBuffer(inQue, 2, TILE * sizeof(float));
        pipe.InitBuffer(outQue, 2, TILE * sizeof(float));
        pipe.InitBuffer(tmpBuf, TILE * sizeof(float));

        uint64_t total = (uint64_t)M * N;
        uint32_t blockIdx = GetBlockIdx();
        uint32_t numBlocks = GetBlockNum();

        uint64_t perBlock = (total + numBlocks - 1) / numBlocks;
        perBlock = (perBlock + 7) / 8 * 8;

        uint64_t start = (uint64_t)blockIdx * perBlock;
        if (start >= total) return;
        uint64_t end = (start + perBlock > total) ? total : (start + perBlock);
        uint64_t myElements = end - start;

        for (uint64_t offset = 0; offset < myElements; offset += TILE) {
            uint32_t curLen = (myElements - offset < TILE) ? (uint32_t)(myElements - offset) : TILE;

            LocalTensor<float> yInAlloc = inQue.AllocTensor<float>();
            DataCopyExtParams copyInParams;
            copyInParams.blockCount = 1;
            copyInParams.blockLen = curLen * (uint32_t)sizeof(float);
            copyInParams.srcStride = 0;
            copyInParams.dstStride = 0;
            copyInParams.rsv = 0;
            DataCopyPadExtParams<float> padParams;
            padParams.isPad = false;
            padParams.leftPadding = 0;
            padParams.rightPadding = 0;
            padParams.paddingValue = 0;
            DataCopyPad(yInAlloc, yGm[start + offset], copyInParams, padParams);
            inQue.EnQue(yInAlloc);

            LocalTensor<float> yLocal = inQue.DeQue<float>();
            LocalTensor<float> tmp = tmpBuf.Get<float>();
            LocalTensor<float> yOut = outQue.AllocTensor<float>();

            Sigmoid<float>(tmp, yLocal, curLen);
            Muls<float>(tmp, tmp, scalingFactor, curLen);
            Add<float>(yOut, yLocal, tmp, curLen);

            outQue.EnQue(yOut);
            inQue.FreeTensor(yLocal);

            LocalTensor<float> yRes = outQue.DeQue<float>();
            DataCopyExtParams copyOutParams;
            copyOutParams.blockCount = 1;
            copyOutParams.blockLen = curLen * (uint32_t)sizeof(float);
            copyOutParams.srcStride = 0;
            copyOutParams.dstStride = 0;
            copyOutParams.rsv = 0;
            DataCopyPad(yGm[start + offset], yRes, copyOutParams);
            outQue.FreeTensor(yRes);
        }
    }

public:
    using A_TYPE = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using B_TYPE = MatmulType<TPosition::GM, CubeFormat::ND, float, true>;
    using BIAS_TYPE = MatmulType<TPosition::GM, CubeFormat::ND, float>;
    using C_TYPE = MatmulType<TPosition::GM, CubeFormat::ND, float>;

    Matmul<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> mm;
    TPipe pipe;

private:
    GlobalTensor<float> xGm, weightGm, biasGm, yGm;
    TQue<TPosition::VECIN, 2> inQue;
    TQue<TPosition::VECOUT, 2> outQue;
    TBuf<TPosition::VECCALC> tmpBuf;
    uint32_t M, N, K;
    float scalingFactor;
};

extern "C" __global__ __aicore__ void gemm_sigmoid_scaling_residual_add_custom(
    GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    KernelGemmSigmoidScalingResidualAdd op;
    REGIST_MATMUL_OBJ(&op.pipe, GetSysWorkSpacePtr(), op.mm, &tilingData.cubeTilingData);
    op.Init(x, weight, bias, y, tilingData.M, tilingData.N, tilingData.K, tilingData.scalingFactor);

    if ASCEND_IS_AIC {
        op.ProcessMatmul();
    }

    SyncAll();

    if ASCEND_IS_AIV {
        op.ProcessVector();
    }
}
