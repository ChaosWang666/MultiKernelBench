
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(Conv3dReluLeakyReluGeluSigmoidBiasAddCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalTasks);
  TILING_DATA_FIELD_DEF(uint32_t, channels);
  TILING_DATA_FIELD_DEF(uint32_t, elemsPerChannel);
  TILING_DATA_FIELD_DEF(uint32_t, tileLen);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Conv3dReluLeakyReluGeluSigmoidBiasAddCustom, Conv3dReluLeakyReluGeluSigmoidBiasAddCustomTilingData)
}
