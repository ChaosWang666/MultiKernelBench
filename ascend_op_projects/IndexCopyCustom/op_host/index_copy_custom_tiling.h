
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(IndexCopyCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, xRows);
  TILING_DATA_FIELD_DEF(uint32_t, cols);
  TILING_DATA_FIELD_DEF(uint32_t, srcRows);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(IndexCopyCustom, IndexCopyCustomTilingData)
}
