
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GemmDivideSumScalingCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batch_size);
  TILING_DATA_FIELD_DEF(uint32_t, input_size);
  TILING_DATA_FIELD_DEF(uint32_t, hidden_size);
  TILING_DATA_FIELD_DEF(float, scaling_factor);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GemmDivideSumScalingCustom, GemmDivideSumScalingCustomTilingData)
}
