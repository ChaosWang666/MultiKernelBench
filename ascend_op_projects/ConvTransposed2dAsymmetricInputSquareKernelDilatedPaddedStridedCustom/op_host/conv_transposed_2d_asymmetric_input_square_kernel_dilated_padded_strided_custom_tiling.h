
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ConvTransposed2dAsymmetricInputSquareKernelDilatedPaddedStridedCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);
  TILING_DATA_FIELD_DEF(uint32_t, inChannels);
  TILING_DATA_FIELD_DEF(uint32_t, outChannels);
  TILING_DATA_FIELD_DEF(uint32_t, heightIn);
  TILING_DATA_FIELD_DEF(uint32_t, widthIn);
  TILING_DATA_FIELD_DEF(uint32_t, heightOut);
  TILING_DATA_FIELD_DEF(uint32_t, widthOut);
  TILING_DATA_FIELD_DEF(uint32_t, kernelSize);
  TILING_DATA_FIELD_DEF(uint32_t, stride);
  TILING_DATA_FIELD_DEF(uint32_t, padding);
  TILING_DATA_FIELD_DEF(uint32_t, dilation);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ConvTransposed2dAsymmetricInputSquareKernelDilatedPaddedStridedCustom, ConvTransposed2dAsymmetricInputSquareKernelDilatedPaddedStridedCustomTilingData)
}
