
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AveragePooling2dCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);
  TILING_DATA_FIELD_DEF(uint32_t, channels);
  TILING_DATA_FIELD_DEF(uint32_t, inputHeight);
  TILING_DATA_FIELD_DEF(uint32_t, inputWidth);
  TILING_DATA_FIELD_DEF(uint32_t, outputHeight);
  TILING_DATA_FIELD_DEF(uint32_t, outputWidth);
  TILING_DATA_FIELD_DEF(uint32_t, kernelSize);
  TILING_DATA_FIELD_DEF(uint32_t, stride);
  TILING_DATA_FIELD_DEF(uint32_t, padding);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AveragePooling2dCustom, AveragePooling2dCustomTilingData)
}
