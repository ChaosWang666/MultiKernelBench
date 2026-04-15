
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MatmulAvgPoolGeluScaleMaxCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);
  TILING_DATA_FIELD_DEF(uint32_t, inputLength);
  TILING_DATA_FIELD_DEF(uint32_t, poolKernelSize);
  TILING_DATA_FIELD_DEF(float, scaleFactor);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MatmulAvgPoolGeluScaleMaxCustom, MatmulAvgPoolGeluScaleMaxCustomTilingData)
}
