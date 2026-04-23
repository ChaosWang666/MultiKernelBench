
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(Conv3dScalingTanhMultiplySigmoidCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalChannels);
  TILING_DATA_FIELD_DEF(uint32_t, perChannelSize);
  TILING_DATA_FIELD_DEF(uint32_t, channels);
  TILING_DATA_FIELD_DEF(uint32_t, channelsPerCore);
  TILING_DATA_FIELD_DEF(uint32_t, tailChannels);
  TILING_DATA_FIELD_DEF(uint32_t, tileLength);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Conv3dScalingTanhMultiplySigmoidCustom, Conv3dScalingTanhMultiplySigmoidCustomTilingData)
}
