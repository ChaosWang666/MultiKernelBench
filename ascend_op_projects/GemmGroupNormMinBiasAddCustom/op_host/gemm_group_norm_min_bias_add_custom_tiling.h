
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GemmGroupNormMinBiasAddCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batch_size);
  TILING_DATA_FIELD_DEF(uint32_t, in_features);
  TILING_DATA_FIELD_DEF(uint32_t, out_features);
  TILING_DATA_FIELD_DEF(uint32_t, num_groups);
  TILING_DATA_FIELD_DEF(uint32_t, bias_shape_0);
  TILING_DATA_FIELD_DEF(uint32_t, bias_shape_1);
  TILING_DATA_FIELD_DEF(uint32_t, bias_shape_2);
  TILING_DATA_FIELD_DEF(uint32_t, bias_shape_3);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GemmGroupNormMinBiasAddCustom, GemmGroupNormMinBiasAddCustomTilingData)
}
