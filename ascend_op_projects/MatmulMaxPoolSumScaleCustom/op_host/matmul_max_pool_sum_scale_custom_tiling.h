
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MatmulMaxPoolSumScaleCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalBatches);
  TILING_DATA_FIELD_DEF(uint32_t, features);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MatmulMaxPoolSumScaleCustom, MatmulMaxPoolSumScaleCustomTilingData)
}
