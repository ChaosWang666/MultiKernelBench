
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GroupNormCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);
  TILING_DATA_FIELD_DEF(uint32_t, numGroups);
  TILING_DATA_FIELD_DEF(uint32_t, numChannels);
  TILING_DATA_FIELD_DEF(uint32_t, numHW);
  TILING_DATA_FIELD_DEF(uint32_t, channelsPerGroup);
  TILING_DATA_FIELD_DEF(uint32_t, groupSize);
  TILING_DATA_FIELD_DEF(float, eps);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GroupNormCustom, GroupNormCustomTilingData)
}
