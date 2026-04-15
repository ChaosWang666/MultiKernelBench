
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(L2NormCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);
  TILING_DATA_FIELD_DEF(uint32_t, dim);
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(L2NormCustom, L2NormCustomTilingData)
}
