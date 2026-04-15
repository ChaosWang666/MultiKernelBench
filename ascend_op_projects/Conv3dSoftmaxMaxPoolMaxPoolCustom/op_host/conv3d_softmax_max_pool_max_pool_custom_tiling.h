
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(Conv3dSoftmaxMaxPoolMaxPoolCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batch);
  TILING_DATA_FIELD_DEF(uint32_t, inChannels);
  TILING_DATA_FIELD_DEF(uint32_t, outChannels);
  TILING_DATA_FIELD_DEF(uint32_t, depth);
  TILING_DATA_FIELD_DEF(uint32_t, height);
  TILING_DATA_FIELD_DEF(uint32_t, width);
  TILING_DATA_FIELD_DEF(uint32_t, kernelSize);
  TILING_DATA_FIELD_DEF(uint32_t, poolKernelSize);
  TILING_DATA_FIELD_DEF(uint32_t, paddedDepth);
  TILING_DATA_FIELD_DEF(uint32_t, paddedHeight);
  TILING_DATA_FIELD_DEF(uint32_t, paddedWidth);
  TILING_DATA_FIELD_DEF(uint32_t, outDepth);
  TILING_DATA_FIELD_DEF(uint32_t, outHeight);
  TILING_DATA_FIELD_DEF(uint32_t, outWidth);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Conv3dSoftmaxMaxPoolMaxPoolCustom, Conv3dSoftmaxMaxPoolMaxPoolCustomTilingData)
}
