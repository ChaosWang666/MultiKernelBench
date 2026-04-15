
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GooglenetInceptionModuleCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, inChannels);
  TILING_DATA_FIELD_DEF(uint32_t, out1x1);
  TILING_DATA_FIELD_DEF(uint32_t, reduce3x3);
  TILING_DATA_FIELD_DEF(uint32_t, out3x3);
  TILING_DATA_FIELD_DEF(uint32_t, reduce5x5);
  TILING_DATA_FIELD_DEF(uint32_t, out5x5);
  TILING_DATA_FIELD_DEF(uint32_t, poolProj);
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);
  TILING_DATA_FIELD_DEF(uint32_t, height);
  TILING_DATA_FIELD_DEF(uint32_t, width);
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GooglenetInceptionModuleCustom, GooglenetInceptionModuleCustomTilingData)
}
