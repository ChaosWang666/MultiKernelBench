
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ConvTranspose3dSumResidualAddMultiplyResidualAddCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalFeatureMaps);
  TILING_DATA_FIELD_DEF(uint32_t, featureMapSize);
  TILING_DATA_FIELD_DEF(uint32_t, channels);
  TILING_DATA_FIELD_DEF(uint32_t, tileLength);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ConvTranspose3dSumResidualAddMultiplyResidualAddCustom, ConvTranspose3dSumResidualAddMultiplyResidualAddCustomTilingData)
}
