
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ConvDepthwiseSeparable2dCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);
  TILING_DATA_FIELD_DEF(uint32_t, inChannels);
  TILING_DATA_FIELD_DEF(uint32_t, outChannels);
  TILING_DATA_FIELD_DEF(uint32_t, height);
  TILING_DATA_FIELD_DEF(uint32_t, width);
  TILING_DATA_FIELD_DEF(uint32_t, kernelSize);
  TILING_DATA_FIELD_DEF(uint32_t, stride);
  TILING_DATA_FIELD_DEF(uint32_t, padding);
  TILING_DATA_FIELD_DEF(uint32_t, dilation);
  TILING_DATA_FIELD_DEF(uint32_t, depthwiseOutHeight);
  TILING_DATA_FIELD_DEF(uint32_t, depthwiseOutWidth);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ConvDepthwiseSeparable2dCustom, ConvDepthwiseSeparable2dCustomTilingData)
}
