
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(Conv3dMultiplyInstanceNormClampMultiplyMaxCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);
  TILING_DATA_FIELD_DEF(uint32_t, inChannels);
  TILING_DATA_FIELD_DEF(uint32_t, outChannels);
  TILING_DATA_FIELD_DEF(uint32_t, depth);
  TILING_DATA_FIELD_DEF(uint32_t, height);
  TILING_DATA_FIELD_DEF(uint32_t, width);
  TILING_DATA_FIELD_DEF(uint32_t, kernelDepth);
  TILING_DATA_FIELD_DEF(uint32_t, kernelHeight);
  TILING_DATA_FIELD_DEF(uint32_t, kernelWidth);
  TILING_DATA_FIELD_DEF(uint32_t, padDepth);
  TILING_DATA_FIELD_DEF(uint32_t, padHeight);
  TILING_DATA_FIELD_DEF(uint32_t, padWidth);
  TILING_DATA_FIELD_DEF(float, clampMin);
  TILING_DATA_FIELD_DEF(float, clampMax);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Conv3dMultiplyInstanceNormClampMultiplyMaxCustom, Conv3dMultiplyInstanceNormClampMultiplyMaxCustomTilingData)
}
