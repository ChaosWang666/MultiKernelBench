
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SumReductionOverADimensionCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, dim);
  TILING_DATA_FIELD_DEF(uint32_t, shape0);
  TILING_DATA_FIELD_DEF(uint32_t, shape1);
  TILING_DATA_FIELD_DEF(uint32_t, shape2);
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SumReductionOverADimensionCustom, SumReductionOverADimensionCustomTilingData)
}
