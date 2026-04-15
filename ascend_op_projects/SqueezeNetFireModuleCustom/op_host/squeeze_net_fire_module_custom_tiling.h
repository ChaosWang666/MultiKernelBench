
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SqueezeNetFireModuleCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);
  TILING_DATA_FIELD_DEF(uint32_t, inChannels);
  TILING_DATA_FIELD_DEF(uint32_t, outChannels);
  TILING_DATA_FIELD_DEF(uint32_t, height);
  TILING_DATA_FIELD_DEF(uint32_t, width);
  TILING_DATA_FIELD_DEF(uint32_t, squeezeChannels);
  TILING_DATA_FIELD_DEF(uint32_t, expand1x1Channels);
  TILING_DATA_FIELD_DEF(uint32_t, expand3x3Channels);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SqueezeNetFireModuleCustom, SqueezeNetFireModuleCustomTilingData)
}
