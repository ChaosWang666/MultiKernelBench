
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ConvStandard3dSquareInputAsymmetricKernelCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);
  TILING_DATA_FIELD_DEF(uint32_t, inChannels);
  TILING_DATA_FIELD_DEF(uint32_t, outChannels);
  TILING_DATA_FIELD_DEF(uint32_t, kernelWidth);
  TILING_DATA_FIELD_DEF(uint32_t, kernelHeight);
  TILING_DATA_FIELD_DEF(uint32_t, kernelDepth);
  TILING_DATA_FIELD_DEF(uint32_t, strideW);
  TILING_DATA_FIELD_DEF(uint32_t, strideH);
  TILING_DATA_FIELD_DEF(uint32_t, strideD);
  TILING_DATA_FIELD_DEF(uint32_t, padW);
  TILING_DATA_FIELD_DEF(uint32_t, padH);
  TILING_DATA_FIELD_DEF(uint32_t, padD);
  TILING_DATA_FIELD_DEF(uint32_t, dilationW);
  TILING_DATA_FIELD_DEF(uint32_t, dilationH);
  TILING_DATA_FIELD_DEF(uint32_t, dilationD);
  TILING_DATA_FIELD_DEF(uint32_t, inputWidth);
  TILING_DATA_FIELD_DEF(uint32_t, inputHeight);
  TILING_DATA_FIELD_DEF(uint32_t, inputDepth);
  TILING_DATA_FIELD_DEF(uint32_t, outputWidth);
  TILING_DATA_FIELD_DEF(uint32_t, outputHeight);
  TILING_DATA_FIELD_DEF(uint32_t, outputDepth);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ConvStandard3dSquareInputAsymmetricKernelCustom, ConvStandard3dSquareInputAsymmetricKernelCustomTilingData)
}
