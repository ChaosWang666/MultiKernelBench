
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(Conv2dInstanceNormDivideCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalInstances);
  TILING_DATA_FIELD_DEF(uint32_t, hw);
  TILING_DATA_FIELD_DEF(uint32_t, tileSize);
  TILING_DATA_FIELD_DEF(uint32_t, instancesPerCore);
  TILING_DATA_FIELD_DEF(float, eps);
  TILING_DATA_FIELD_DEF(float, divideBy);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Conv2dInstanceNormDivideCustom, Conv2dInstanceNormDivideCustomTilingData)
}
