
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ConvTranspose3dScalingAvgPoolBiasAddScalingCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, inChannels);
  TILING_DATA_FIELD_DEF(uint32_t, outChannels);
  TILING_DATA_FIELD_DEF(uint32_t, kernelSize);
  TILING_DATA_FIELD_DEF(uint32_t, stride);
  TILING_DATA_FIELD_DEF(uint32_t, padding);
  TILING_DATA_FIELD_DEF(float, scale1);
  TILING_DATA_FIELD_DEF(float, scale2);
  TILING_DATA_FIELD_DEF(uint32_t, biasShape0);
  TILING_DATA_FIELD_DEF(uint32_t, biasShape1);
  TILING_DATA_FIELD_DEF(uint32_t, biasShape2);
  TILING_DATA_FIELD_DEF(uint32_t, biasShape3);
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ConvTranspose3dScalingAvgPoolBiasAddScalingCustom, ConvTranspose3dScalingAvgPoolBiasAddScalingCustomTilingData)
}
