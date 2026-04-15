
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ConvTranspose3dSumLayerNormAvgPoolGeluCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);
  TILING_DATA_FIELD_DEF(uint32_t, inChannels);
  TILING_DATA_FIELD_DEF(uint32_t, outChannels);
  TILING_DATA_FIELD_DEF(uint32_t, depth);
  TILING_DATA_FIELD_DEF(uint32_t, height);
  TILING_DATA_FIELD_DEF(uint32_t, width);
  TILING_DATA_FIELD_DEF(uint32_t, kernelDepth);
  TILING_DATA_FIELD_DEF(uint32_t, kernelHeight);
  TILING_DATA_FIELD_DEF(uint32_t, kernelWidth);
  TILING_DATA_FIELD_DEF(uint32_t, strideDepth);
  TILING_DATA_FIELD_DEF(uint32_t, strideHeight);
  TILING_DATA_FIELD_DEF(uint32_t, strideWidth);
  TILING_DATA_FIELD_DEF(uint32_t, padDepth);
  TILING_DATA_FIELD_DEF(uint32_t, padHeight);
  TILING_DATA_FIELD_DEF(uint32_t, padWidth);
  TILING_DATA_FIELD_DEF(uint32_t, outPadDepth);
  TILING_DATA_FIELD_DEF(uint32_t, outPadHeight);
  TILING_DATA_FIELD_DEF(uint32_t, outPadWidth);
  TILING_DATA_FIELD_DEF(uint32_t, poolKernelDepth);
  TILING_DATA_FIELD_DEF(uint32_t, poolKernelHeight);
  TILING_DATA_FIELD_DEF(uint32_t, poolKernelWidth);
  TILING_DATA_FIELD_DEF(float, sumWeight);
  TILING_DATA_FIELD_DEF(uint32_t, normShape);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ConvTranspose3dSumLayerNormAvgPoolGeluCustom, ConvTranspose3dSumLayerNormAvgPoolGeluCustomTilingData)
}
