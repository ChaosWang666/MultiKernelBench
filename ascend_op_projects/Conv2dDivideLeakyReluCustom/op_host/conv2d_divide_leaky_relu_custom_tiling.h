
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(Conv2dDivideLeakyReluCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);
  TILING_DATA_FIELD_DEF(uint32_t, inChannels);
  TILING_DATA_FIELD_DEF(uint32_t, outChannels);
  TILING_DATA_FIELD_DEF(uint32_t, height);
  TILING_DATA_FIELD_DEF(uint32_t, width);
  TILING_DATA_FIELD_DEF(uint32_t, kernelH);
  TILING_DATA_FIELD_DEF(uint32_t, kernelW);
  TILING_DATA_FIELD_DEF(float, divisor);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Conv2dDivideLeakyReluCustom, Conv2dDivideLeakyReluCustomTilingData)
}
