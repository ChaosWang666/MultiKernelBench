
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ConvDepthwise2dSquareInputSquareKernelCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batch);
  TILING_DATA_FIELD_DEF(uint32_t, inChannels);
  TILING_DATA_FIELD_DEF(uint32_t, outHeight);
  TILING_DATA_FIELD_DEF(uint32_t, outWidth);
  TILING_DATA_FIELD_DEF(uint32_t, kernelH);
  TILING_DATA_FIELD_DEF(uint32_t, kernelW);
  TILING_DATA_FIELD_DEF(uint32_t, strideH);
  TILING_DATA_FIELD_DEF(uint32_t, strideW);
  TILING_DATA_FIELD_DEF(uint32_t, padH);
  TILING_DATA_FIELD_DEF(uint32_t, padW);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ConvDepthwise2dSquareInputSquareKernelCustom, ConvDepthwise2dSquareInputSquareKernelCustomTilingData)
}
