
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(Conv2dScalingMinCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);
  TILING_DATA_FIELD_DEF(uint32_t, channels);
  TILING_DATA_FIELD_DEF(uint32_t, spatialSize);
  TILING_DATA_FIELD_DEF(uint32_t, tileLen);
  TILING_DATA_FIELD_DEF(float, scaleFactor);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Conv2dScalingMinCustom, Conv2dScalingMinCustomTilingData)
}
