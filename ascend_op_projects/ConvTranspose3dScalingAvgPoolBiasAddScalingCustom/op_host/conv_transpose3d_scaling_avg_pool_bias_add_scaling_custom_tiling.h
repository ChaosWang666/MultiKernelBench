
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ConvTranspose3dScalingAvgPoolBiasAddScalingCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalOuter);
  TILING_DATA_FIELD_DEF(uint32_t, numChannels);
  TILING_DATA_FIELD_DEF(uint32_t, spatial);
  TILING_DATA_FIELD_DEF(uint32_t, tileSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ConvTranspose3dScalingAvgPoolBiasAddScalingCustom, ConvTranspose3dScalingAvgPoolBiasAddScalingCustomTilingData)
}
