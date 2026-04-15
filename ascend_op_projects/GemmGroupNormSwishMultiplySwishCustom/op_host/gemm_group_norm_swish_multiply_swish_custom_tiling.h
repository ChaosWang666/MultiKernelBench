
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GemmGroupNormSwishMultiplySwishCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batch_size);
  TILING_DATA_FIELD_DEF(uint32_t, in_features);
  TILING_DATA_FIELD_DEF(uint32_t, out_features);
  TILING_DATA_FIELD_DEF(uint32_t, num_groups);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GemmGroupNormSwishMultiplySwishCustom, GemmGroupNormSwishMultiplySwishCustomTilingData)
}
