
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GemmMultiplyLeakyreluCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);
  TILING_DATA_FIELD_DEF(float, multiplier);
  TILING_DATA_FIELD_DEF(float, negSlope);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GemmMultiplyLeakyreluCustom, GemmMultiplyLeakyreluCustomTilingData)
}
