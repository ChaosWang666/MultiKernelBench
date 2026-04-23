
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MatmulScaleResidualAddClampLogSumExpMishCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);
  TILING_DATA_FIELD_DEF(uint32_t, hiddenSize);
  TILING_DATA_FIELD_DEF(float, scaleFactor);
  TILING_DATA_FIELD_DEF(float, clampMin);
  TILING_DATA_FIELD_DEF(float, clampMax);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MatmulScaleResidualAddClampLogSumExpMishCustom, MatmulScaleResidualAddClampLogSumExpMishCustomTilingData)
}
