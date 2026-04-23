
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GemmGroupNormMinBiasAddCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalRows);
  TILING_DATA_FIELD_DEF(uint32_t, totalCols);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GemmGroupNormMinBiasAddCustom, GemmGroupNormMinBiasAddCustomTilingData)
}
