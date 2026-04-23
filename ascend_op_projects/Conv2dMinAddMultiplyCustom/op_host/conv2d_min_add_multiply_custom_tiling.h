
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(Conv2dMinAddMultiplyCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalPairs);
  TILING_DATA_FIELD_DEF(uint32_t, spatialSize);
  TILING_DATA_FIELD_DEF(uint32_t, channels);
  TILING_DATA_FIELD_DEF(uint32_t, tileSize);
  TILING_DATA_FIELD_DEF(float, constantValue);
  TILING_DATA_FIELD_DEF(float, scalingFactor);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Conv2dMinAddMultiplyCustom, Conv2dMinAddMultiplyCustomTilingData)
}
