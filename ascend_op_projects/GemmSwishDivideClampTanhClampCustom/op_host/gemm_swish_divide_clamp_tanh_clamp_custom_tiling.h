
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GemmSwishDivideClampTanhClampCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batch);
  TILING_DATA_FIELD_DEF(uint32_t, inFeatures);
  TILING_DATA_FIELD_DEF(uint32_t, outFeatures);
  TILING_DATA_FIELD_DEF(bool, hasBias);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GemmSwishDivideClampTanhClampCustom, GemmSwishDivideClampTanhClampCustomTilingData)
}
