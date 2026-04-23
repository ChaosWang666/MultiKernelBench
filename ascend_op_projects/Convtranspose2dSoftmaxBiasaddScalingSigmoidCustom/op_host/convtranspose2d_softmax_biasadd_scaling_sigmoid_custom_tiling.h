
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(Convtranspose2dSoftmaxBiasaddScalingSigmoidCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalNC);
  TILING_DATA_FIELD_DEF(uint32_t, C);
  TILING_DATA_FIELD_DEF(uint32_t, HW);
  TILING_DATA_FIELD_DEF(uint32_t, tileSize);
  TILING_DATA_FIELD_DEF(float, scalingFactor);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Convtranspose2dSoftmaxBiasaddScalingSigmoidCustom, Convtranspose2dSoftmaxBiasaddScalingSigmoidCustomTilingData)
}
