
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ConvTranspose2dMaxPoolHardtanhMeanTanhCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);
  TILING_DATA_FIELD_DEF(uint32_t, inChannels);
  TILING_DATA_FIELD_DEF(uint32_t, outChannels);
  TILING_DATA_FIELD_DEF(uint32_t, height);
  TILING_DATA_FIELD_DEF(uint32_t, width);
  TILING_DATA_FIELD_DEF(uint32_t, kernelSize);
  TILING_DATA_FIELD_DEF(uint32_t, stride);
  TILING_DATA_FIELD_DEF(uint32_t, padding);
  TILING_DATA_FIELD_DEF(uint32_t, maxpoolKernelSize);
  TILING_DATA_FIELD_DEF(uint32_t, maxpoolStride);
  TILING_DATA_FIELD_DEF(float, hardtanhMin);
  TILING_DATA_FIELD_DEF(float, hardtanhMax);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ConvTranspose2dMaxPoolHardtanhMeanTanhCustom, ConvTranspose2dMaxPoolHardtanhMeanTanhCustomTilingData)
}
