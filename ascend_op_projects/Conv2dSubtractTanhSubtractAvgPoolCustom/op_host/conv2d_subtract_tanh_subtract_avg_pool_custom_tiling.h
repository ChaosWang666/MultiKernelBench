
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(Conv2dSubtractTanhSubtractAvgPoolCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);
  TILING_DATA_FIELD_DEF(uint32_t, inChannels);
  TILING_DATA_FIELD_DEF(uint32_t, outChannels);
  TILING_DATA_FIELD_DEF(uint32_t, height);
  TILING_DATA_FIELD_DEF(uint32_t, width);
  TILING_DATA_FIELD_DEF(uint32_t, kernelSize);
  TILING_DATA_FIELD_DEF(float, subtract1Value);
  TILING_DATA_FIELD_DEF(float, subtract2Value);
  TILING_DATA_FIELD_DEF(uint32_t, poolKernelSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Conv2dSubtractTanhSubtractAvgPoolCustom, Conv2dSubtractTanhSubtractAvgPoolCustomTilingData)
}
