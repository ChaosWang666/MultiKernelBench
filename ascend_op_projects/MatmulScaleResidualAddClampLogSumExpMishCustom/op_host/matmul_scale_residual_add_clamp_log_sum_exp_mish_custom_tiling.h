
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MatmulScaleResidualAddClampLogSumExpMishCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);
  TILING_DATA_FIELD_DEF(float, scale_factor);
  TILING_DATA_FIELD_DEF(float, clamp_min);
  TILING_DATA_FIELD_DEF(float, clamp_max);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MatmulScaleResidualAddClampLogSumExpMishCustom, MatmulScaleResidualAddClampLogSumExpMishCustomTilingData)
}
