
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ConvTransposed1dAsymmetricInputSquareKernelPaddedStridedDilatedCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batch);
  TILING_DATA_FIELD_DEF(uint32_t, inChannels);
  TILING_DATA_FIELD_DEF(uint32_t, outChannels);
  TILING_DATA_FIELD_DEF(uint32_t, kernelSize);
  TILING_DATA_FIELD_DEF(uint32_t, stride);
  TILING_DATA_FIELD_DEF(uint32_t, padding);
  TILING_DATA_FIELD_DEF(uint32_t, dilation);
  TILING_DATA_FIELD_DEF(uint32_t, inputLength);
  TILING_DATA_FIELD_DEF(uint32_t, outputLength);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ConvTransposed1dAsymmetricInputSquareKernelPaddedStridedDilatedCustom, ConvTransposed1dAsymmetricInputSquareKernelPaddedStridedDilatedCustomTilingData)
}
