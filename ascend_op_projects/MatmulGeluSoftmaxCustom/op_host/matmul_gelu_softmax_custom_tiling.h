
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MatmulGeluSoftmaxCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, rows);
  TILING_DATA_FIELD_DEF(uint32_t, cols);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MatmulGeluSoftmaxCustom, MatmulGeluSoftmaxCustomTilingData)
}
