
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ScatterAddCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, numRows);
  TILING_DATA_FIELD_DEF(uint32_t, xCols);
  TILING_DATA_FIELD_DEF(uint32_t, idxCols);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ScatterAddCustom, ScatterAddCustomTilingData)
}
