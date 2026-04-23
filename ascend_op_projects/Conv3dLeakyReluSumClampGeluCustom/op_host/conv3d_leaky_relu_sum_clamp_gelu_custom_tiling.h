
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(Conv3dLeakyReluSumClampGeluCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalRows);
  TILING_DATA_FIELD_DEF(uint32_t, channelSize);
  TILING_DATA_FIELD_DEF(uint32_t, spatialSize);
  TILING_DATA_FIELD_DEF(uint32_t, rowTileSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Conv3dLeakyReluSumClampGeluCustom, Conv3dLeakyReluSumClampGeluCustomTilingData)
}
