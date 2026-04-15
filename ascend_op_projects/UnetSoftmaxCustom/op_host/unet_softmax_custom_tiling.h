
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(UnetSoftmaxCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, blockSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(UnetSoftmaxCustom, UnetSoftmaxCustomTilingData)
}
