
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(InplaceUpdateCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, numRows);
  TILING_DATA_FIELD_DEF(uint32_t, rowLength);
  TILING_DATA_FIELD_DEF(uint32_t, numIdx);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(InplaceUpdateCustom, InplaceUpdateCustomTilingData)
}
