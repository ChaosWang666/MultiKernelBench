
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;
using namespace matmul;

constexpr uint32_t VECTOR_TILE_SIZE = 1024;

class KernelGemmReluDivide {
public:
    __aicore__ inline KernelGemmReluDivide() {}

    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c,
                                 const TCubeTiling& tiling_, float divisor_, TPipe* pipe_)
    {
        this->tiling = tiling_;
        this->divisor = divisor_;
        this->pipe = pipe_;
        aGm.SetGlobalBuffer((__gm__ float*)a);
        bGm.SetGlobalBuffer((__gm__ float*)b);
        biasGm.SetGlobalBuffer((__gm__ float*)bias);
        cGm.SetGlobalBuffer((__gm__ float*)c);

        pipe->InitBuffer(inQ, 2, VECTOR_TILE_SIZE * sizeof(float));
        pipe->InitBuffer(outQ, 2, VECTOR_TILE_SIZE * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        int coreId = GetBlockIdx();
        int64_t M = tiling.M;
        int64_t N = tiling.N;
        int64_t K = tiling.K;
        int64_t singleCoreM = tiling.singleCoreM;
        int64_t singleCoreN = tiling.singleCoreN;

        int64_t nBlocks = (N + singleCoreN - 1) / singleCoreN;
        int64_t mBlocks = (M + singleCoreM - 1) / singleCoreM;
        int64_t totalBlocks = nBlocks * mBlocks;

        if (coreId >= totalBlocks) {
            return;
        }

        int64_t blockIdxM = coreId / nBlocks;
        int64_t blockIdxN = coreId % nBlocks;

        int64_t offsetM = blockIdxM * singleCoreM;
        int64_t offsetN = blockIdxN * singleCoreN;
        int64_t curM = (offsetM + singleCoreM > M) ? (M - offsetM) : singleCoreM;
        int64_t curN = (offsetN + singleCoreN > N) ? (N - offsetN) : singleCoreN;

        if (curM <= 0 || curN <= 0) {
            return;
        }

        mm.SetTensorA(aGm[offsetM * K]);
        mm.SetTensorB(bGm[offsetN * K]);
        mm.SetBias(biasGm[offsetN]);
        mm.SetTail(curM, curN);
        mm.IterateAll(cGm[offsetM * N + offsetN]);
        mm.End();

        float invDivisor = 1.0f / divisor;

        for (int64_t m = 0; m < curM; m++) {
            int64_t globalRow = offsetM + m;
            for (int64_t colStart = 0; colStart < curN; colStart += VECTOR_TILE_SIZE) {
                int64_t tileLen = (colStart + (int64_t)VECTOR_TILE_SIZE > curN) ? (curN - colStart) : (int64_t)VECTOR_TILE_SIZE;
                int64_t gmOffset = globalRow * N + offsetN + colStart;

                LocalTensor<float> inLocal = inQ.AllocTensor<float>();
                DataCopyExtParams copyParams;
                copyParams.blockCount = 1;
                copyParams.blockLen = (uint32_t)(tileLen * sizeof(float));
                copyParams.srcStride = 0;
                copyParams.dstStride = 0;
                copyParams.rsv = 0;
                DataCopyPadExtParams<float> padParams;
                padParams.isPad = false;
                padParams.leftPadding = 0;
                padParams.rightPadding = 0;
                padParams.paddingValue = 0.0f;
                DataCopyPad(inLocal, cGm[gmOffset], copyParams, padParams);
                inQ.EnQue(inLocal);

                LocalTensor<float> inData = inQ.DeQue<float>();
                LocalTensor<float> outLocal = outQ.AllocTensor<float>();
                Relu(outLocal, inData, (int32_t)tileLen);
                Muls(outLocal, outLocal, invDivisor, (int32_t)tileLen);
                outQ.EnQue(outLocal);
                inQ.FreeTensor(inData);

                LocalTensor<float> outData = outQ.DeQue<float>();
                DataCopyPad(cGm[gmOffset], outData, copyParams);
                outQ.FreeTensor(outData);
            }
        }
    }

    Matmul<MatmulType<TPosition::GM, CubeFormat::ND, float>,
           MatmulType<TPosition::GM, CubeFormat::ND, float, true>,
           MatmulType<TPosition::GM, CubeFormat::ND, float>,
           MatmulType<TPosition::GM, CubeFormat::ND, float>> mm;

private:
    TQue<TPosition::VECIN, 2> inQ;
    TQue<TPosition::VECOUT, 2> outQ;
    GlobalTensor<float> aGm;
    GlobalTensor<float> bGm;
    GlobalTensor<float> biasGm;
    GlobalTensor<float> cGm;
    TCubeTiling tiling;
    float divisor;
    TPipe* pipe;
};

extern "C" __global__ __aicore__ void gemm_relu_divide_custom(GM_ADDR x, GM_ADDR weight, GM_ADDR bias, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    KernelGemmReluDivide op;
    op.Init(x, weight, bias, y, tilingData.cubeTilingData, tilingData.divisor, &pipe);
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), op.mm, &tilingData.cubeTilingData);
    op.Process();
}
