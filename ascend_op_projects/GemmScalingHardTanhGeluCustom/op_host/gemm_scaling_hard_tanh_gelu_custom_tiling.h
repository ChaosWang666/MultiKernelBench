
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GemmScalingHardTanhGeluCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);
  TILING_DATA_FIELD_DEF(float, scalingFactor);
  TILING_DATA_FIELD_DEF(float, hardtanhMin);
  TILING_DATA_FIELD_DEF(float, hardtanhMax);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GemmScalingHardTanhGeluCustom, GemmScalingHardTanhGeluCustomTilingData)
}
