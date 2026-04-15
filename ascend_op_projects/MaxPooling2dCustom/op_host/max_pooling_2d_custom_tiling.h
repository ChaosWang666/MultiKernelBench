
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MaxPooling2dCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);
  TILING_DATA_FIELD_DEF(uint32_t, channels);
  TILING_DATA_FIELD_DEF(uint32_t, height);
  TILING_DATA_FIELD_DEF(uint32_t, width);
  TILING_DATA_FIELD_DEF(uint32_t, kernelSize);
  TILING_DATA_FIELD_DEF(uint32_t, stride);
  TILING_DATA_FIELD_DEF(uint32_t, padding);
  TILING_DATA_FIELD_DEF(uint32_t, outHeight);
  TILING_DATA_FIELD_DEF(uint32_t, outWidth);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MaxPooling2dCustom, MaxPooling2dCustomTilingData)
}
