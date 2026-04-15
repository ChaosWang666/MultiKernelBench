
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(Conv2dMinTanhTanhCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);
  TILING_DATA_FIELD_DEF(uint32_t, channels);
  TILING_DATA_FIELD_DEF(uint32_t, spatialSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Conv2dMinTanhTanhCustom, Conv2dMinTanhTanhCustomTilingData)
}
