
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(Conv2dMinAddMultiplyCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);
  TILING_DATA_FIELD_DEF(float, constantValue);
  TILING_DATA_FIELD_DEF(float, scalingFactor);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Conv2dMinAddMultiplyCustom, Conv2dMinAddMultiplyCustomTilingData)
}
