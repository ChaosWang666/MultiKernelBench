
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ConvTranspose3dLeakyReluMultiplyLeakyReluMaxCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalChannels);
  TILING_DATA_FIELD_DEF(uint32_t, channelSize);
  TILING_DATA_FIELD_DEF(uint32_t, channels);
  TILING_DATA_FIELD_DEF(uint32_t, tileLen);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ConvTranspose3dLeakyReluMultiplyLeakyReluMaxCustom, ConvTranspose3dLeakyReluMultiplyLeakyReluMaxCustomTilingData)
}
