
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ShufflenetUnitCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);
  TILING_DATA_FIELD_DEF(uint32_t, inChannels);
  TILING_DATA_FIELD_DEF(uint32_t, outChannels);
  TILING_DATA_FIELD_DEF(uint32_t, height);
  TILING_DATA_FIELD_DEF(uint32_t, width);
  TILING_DATA_FIELD_DEF(uint32_t, groups);
  TILING_DATA_FIELD_DEF(uint32_t, midChannels);
  TILING_DATA_FIELD_DEF(uint32_t, conv1WeightSize);
  TILING_DATA_FIELD_DEF(uint32_t, conv2WeightSize);
  TILING_DATA_FIELD_DEF(uint32_t, conv3WeightSize);
  TILING_DATA_FIELD_DEF(uint32_t, shortcutConvWeightSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ShufflenetUnitCustom, ShufflenetUnitCustomTilingData)
}
