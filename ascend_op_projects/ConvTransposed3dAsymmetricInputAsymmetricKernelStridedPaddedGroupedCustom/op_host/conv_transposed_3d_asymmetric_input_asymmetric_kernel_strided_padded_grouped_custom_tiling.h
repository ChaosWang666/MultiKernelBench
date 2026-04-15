
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ConvTransposed3dAsymmetricInputAsymmetricKernelStridedPaddedGroupedCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batch);
  TILING_DATA_FIELD_DEF(uint32_t, inChannels);
  TILING_DATA_FIELD_DEF(uint32_t, outChannels);
  TILING_DATA_FIELD_DEF(uint32_t, kernelDepth);
  TILING_DATA_FIELD_DEF(uint32_t, kernelHeight);
  TILING_DATA_FIELD_DEF(uint32_t, kernelWidth);
  TILING_DATA_FIELD_DEF(uint32_t, strideDepth);
  TILING_DATA_FIELD_DEF(uint32_t, strideHeight);
  TILING_DATA_FIELD_DEF(uint32_t, strideWidth);
  TILING_DATA_FIELD_DEF(uint32_t, padDepth);
  TILING_DATA_FIELD_DEF(uint32_t, padHeight);
  TILING_DATA_FIELD_DEF(uint32_t, padWidth);
  TILING_DATA_FIELD_DEF(uint32_t, outDepth);
  TILING_DATA_FIELD_DEF(uint32_t, outHeight);
  TILING_DATA_FIELD_DEF(uint32_t, outWidth);
  TILING_DATA_FIELD_DEF(uint32_t, groups);
  TILING_DATA_FIELD_DEF(uint32_t, totalElements);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ConvTransposed3dAsymmetricInputAsymmetricKernelStridedPaddedGroupedCustom, ConvTransposed3dAsymmetricInputAsymmetricKernelStridedPaddedGroupedCustomTilingData)
}
