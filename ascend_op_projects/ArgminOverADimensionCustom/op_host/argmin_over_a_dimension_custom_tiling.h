
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ArgminOverADimensionCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, dimBefore);
  TILING_DATA_FIELD_DEF(uint32_t, dimSize);
  TILING_DATA_FIELD_DEF(uint32_t, dimAfter);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ArgminOverADimensionCustom, ArgminOverADimensionCustomTilingData)
}
