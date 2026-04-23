
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ConvTranspose3dSumLayerNormAvgPoolGeluCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);
  TILING_DATA_FIELD_DEF(float, sumWeight);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ConvTranspose3dSumLayerNormAvgPoolGeluCustom, ConvTranspose3dSumLayerNormAvgPoolGeluCustomTilingData)
}
