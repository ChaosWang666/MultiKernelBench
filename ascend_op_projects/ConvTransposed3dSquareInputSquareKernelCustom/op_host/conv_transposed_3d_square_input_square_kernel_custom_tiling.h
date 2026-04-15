
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ConvTransposed3dSquareInputSquareKernelCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batch);
  TILING_DATA_FIELD_DEF(uint32_t, inChannels);
  TILING_DATA_FIELD_DEF(uint32_t, outChannels);
  TILING_DATA_FIELD_DEF(uint32_t, kernelSize);
  TILING_DATA_FIELD_DEF(uint32_t, stride);
  TILING_DATA_FIELD_DEF(uint32_t, padding);
  TILING_DATA_FIELD_DEF(uint32_t, outputPadding);
  TILING_DATA_FIELD_DEF(uint32_t, groups);
  TILING_DATA_FIELD_DEF(uint32_t, depth);
  TILING_DATA_FIELD_DEF(uint32_t, height);
  TILING_DATA_FIELD_DEF(uint32_t, width);
  TILING_DATA_FIELD_DEF(uint32_t, outDepth);
  TILING_DATA_FIELD_DEF(uint32_t, outHeight);
  TILING_DATA_FIELD_DEF(uint32_t, outWidth);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ConvTransposed3dSquareInputSquareKernelCustom, ConvTransposed3dSquareInputSquareKernelCustomTilingData)
}
