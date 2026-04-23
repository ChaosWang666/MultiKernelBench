
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(Conv2dMultiplyLeakyReluGeluCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalChannels);
  TILING_DATA_FIELD_DEF(uint32_t, channelSize);
  TILING_DATA_FIELD_DEF(uint32_t, channels);
  TILING_DATA_FIELD_DEF(uint32_t, tileLength);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Conv2dMultiplyLeakyReluGeluCustom, Conv2dMultiplyLeakyReluGeluCustomTilingData)
}
