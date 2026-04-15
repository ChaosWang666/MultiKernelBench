
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ConvDepthwise2dAsymmetricInputAsymmetricKernelCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);
  TILING_DATA_FIELD_DEF(uint32_t, inChannels);
  TILING_DATA_FIELD_DEF(uint32_t, outChannels);
  TILING_DATA_FIELD_DEF(uint32_t, inputHeight);
  TILING_DATA_FIELD_DEF(uint32_t, inputWidth);
  TILING_DATA_FIELD_DEF(uint32_t, kernelHeight);
  TILING_DATA_FIELD_DEF(uint32_t, kernelWidth);
  TILING_DATA_FIELD_DEF(uint32_t, strideH);
  TILING_DATA_FIELD_DEF(uint32_t, strideW);
  TILING_DATA_FIELD_DEF(uint32_t, padH);
  TILING_DATA_FIELD_DEF(uint32_t, padW);
  TILING_DATA_FIELD_DEF(uint32_t, dilationH);
  TILING_DATA_FIELD_DEF(uint32_t, dilationW);
  TILING_DATA_FIELD_DEF(uint32_t, outputHeight);
  TILING_DATA_FIELD_DEF(uint32_t, outputWidth);
  TILING_DATA_FIELD_DEF(uint32_t, tileNumH);
  TILING_DATA_FIELD_DEF(uint32_t, tileNumW);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ConvDepthwise2dAsymmetricInputAsymmetricKernelCustom, ConvDepthwise2dAsymmetricInputAsymmetricKernelCustomTilingData)
}
