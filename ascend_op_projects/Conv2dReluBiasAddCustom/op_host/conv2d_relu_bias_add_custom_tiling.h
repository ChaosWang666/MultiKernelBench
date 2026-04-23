
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(Conv2dReluBiasAddCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalGroups);
  TILING_DATA_FIELD_DEF(uint32_t, elementsPerGroup);
  TILING_DATA_FIELD_DEF(uint32_t, outChannels);
  TILING_DATA_FIELD_DEF(uint32_t, tileLength);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Conv2dReluBiasAddCustom, Conv2dReluBiasAddCustomTilingData)
}
