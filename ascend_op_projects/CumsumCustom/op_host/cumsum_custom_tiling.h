
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(CumsumCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, dim);
  TILING_DATA_FIELD_DEF(uint32_t, dimSize);
  TILING_DATA_FIELD_DEF(uint32_t, outerSize);
  TILING_DATA_FIELD_DEF(uint32_t, innerSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(CumsumCustom, CumsumCustomTilingData)
}
