
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GemmGroupNormHardtanhCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batch_size);
  TILING_DATA_FIELD_DEF(uint32_t, in_features);
  TILING_DATA_FIELD_DEF(uint32_t, out_features);
  TILING_DATA_FIELD_DEF(uint32_t, num_groups);
  TILING_DATA_FIELD_DEF(float, hardtanh_min);
  TILING_DATA_FIELD_DEF(float, hardtanh_max);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GemmGroupNormHardtanhCustom, GemmGroupNormHardtanhCustomTilingData)
}
