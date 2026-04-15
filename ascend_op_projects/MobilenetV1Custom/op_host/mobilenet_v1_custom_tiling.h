
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MobilenetV1CustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);
  TILING_DATA_FIELD_DEF(uint32_t, inputChannels);
  TILING_DATA_FIELD_DEF(uint32_t, height);
  TILING_DATA_FIELD_DEF(uint32_t, width);
  TILING_DATA_FIELD_DEF(uint32_t, outputChannels);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MobilenetV1Custom, MobilenetV1CustomTilingData)
}
