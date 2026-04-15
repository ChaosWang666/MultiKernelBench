
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ConvTransposed2dAsymmetricInputAsymmetricKernelStridedGroupedPaddedDilatedCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);
  TILING_DATA_FIELD_DEF(uint32_t, inChannels);
  TILING_DATA_FIELD_DEF(uint32_t, outChannels);
  TILING_DATA_FIELD_DEF(uint32_t, inHeight);
  TILING_DATA_FIELD_DEF(uint32_t, inWidth);
  TILING_DATA_FIELD_DEF(uint32_t, outHeight);
  TILING_DATA_FIELD_DEF(uint32_t, outWidth);
  TILING_DATA_FIELD_DEF(uint32_t, kernelHeight);
  TILING_DATA_FIELD_DEF(uint32_t, kernelWidth);
  TILING_DATA_FIELD_DEF(uint32_t, strideHeight);
  TILING_DATA_FIELD_DEF(uint32_t, strideWidth);
  TILING_DATA_FIELD_DEF(uint32_t, padHeight);
  TILING_DATA_FIELD_DEF(uint32_t, padWidth);
  TILING_DATA_FIELD_DEF(uint32_t, dilationHeight);
  TILING_DATA_FIELD_DEF(uint32_t, dilationWidth);
  TILING_DATA_FIELD_DEF(uint32_t, groups);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ConvTransposed2dAsymmetricInputAsymmetricKernelStridedGroupedPaddedDilatedCustom, ConvTransposed2dAsymmetricInputAsymmetricKernelStridedGroupedPaddedDilatedCustomTilingData)
}
