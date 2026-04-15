
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GemmScalingHardTanhGeluCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batch);
  TILING_DATA_FIELD_DEF(uint32_t, inFeatures);
  TILING_DATA_FIELD_DEF(uint32_t, outFeatures);
  TILING_DATA_FIELD_DEF(float, scalingFactor);
  TILING_DATA_FIELD_DEF(float, hardTanhMin);
  TILING_DATA_FIELD_DEF(float, hardTanhMax);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GemmScalingHardTanhGeluCustom, GemmScalingHardTanhGeluCustomTilingData)
}
