
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ConvTransposed3dAsymmetricInputAsymmetricKernelCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batch);
  TILING_DATA_FIELD_DEF(uint32_t, inChannels);
  TILING_DATA_FIELD_DEF(uint32_t, outChannels);
  TILING_DATA_FIELD_DEF(uint32_t, depthIn);
  TILING_DATA_FIELD_DEF(uint32_t, heightIn);
  TILING_DATA_FIELD_DEF(uint32_t, widthIn);
  TILING_DATA_FIELD_DEF(uint32_t, depthOut);
  TILING_DATA_FIELD_DEF(uint32_t, heightOut);
  TILING_DATA_FIELD_DEF(uint32_t, widthOut);
  TILING_DATA_FIELD_DEF(uint32_t, kernelDepth);
  TILING_DATA_FIELD_DEF(uint32_t, kernelHeight);
  TILING_DATA_FIELD_DEF(uint32_t, kernelWidth);
  TILING_DATA_FIELD_DEF(uint32_t, strideDepth);
  TILING_DATA_FIELD_DEF(uint32_t, strideHeight);
  TILING_DATA_FIELD_DEF(uint32_t, strideWidth);
  TILING_DATA_FIELD_DEF(uint32_t, padDepth);
  TILING_DATA_FIELD_DEF(uint32_t, padHeight);
  TILING_DATA_FIELD_DEF(uint32_t, padWidth);
  TILING_DATA_FIELD_DEF(uint32_t, outPadDepth);
  TILING_DATA_FIELD_DEF(uint32_t, outPadHeight);
  TILING_DATA_FIELD_DEF(uint32_t, outPadWidth);
  TILING_DATA_FIELD_DEF(uint32_t, groups);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ConvTransposed3dAsymmetricInputAsymmetricKernelCustom, ConvTransposed3dAsymmetricInputAsymmetricKernelCustomTilingData)
}
