
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(Conv3dGroupNormMinClampDropoutCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batch);
  TILING_DATA_FIELD_DEF(uint32_t, inChannels);
  TILING_DATA_FIELD_DEF(uint32_t, outChannels);
  TILING_DATA_FIELD_DEF(uint32_t, depth);
  TILING_DATA_FIELD_DEF(uint32_t, height);
  TILING_DATA_FIELD_DEF(uint32_t, width);
  TILING_DATA_FIELD_DEF(uint32_t, kernelSize);
  TILING_DATA_FIELD_DEF(uint32_t, groups);
  TILING_DATA_FIELD_DEF(float, minValue);
  TILING_DATA_FIELD_DEF(float, maxValue);
  TILING_DATA_FIELD_DEF(float, dropoutP);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Conv3dGroupNormMinClampDropoutCustom, Conv3dGroupNormMinClampDropoutCustomTilingData)
}
