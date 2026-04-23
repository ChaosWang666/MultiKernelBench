
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(Conv3dMaxLogSumExpReluCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batch);
  TILING_DATA_FIELD_DEF(uint32_t, channels);
  TILING_DATA_FIELD_DEF(uint32_t, spatial);
  TILING_DATA_FIELD_DEF(uint32_t, tileLen);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Conv3dMaxLogSumExpReluCustom, Conv3dMaxLogSumExpReluCustomTilingData)
}
