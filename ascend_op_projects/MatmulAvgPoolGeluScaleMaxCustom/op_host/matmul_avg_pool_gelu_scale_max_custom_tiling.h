
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MatmulAvgPoolGeluScaleMaxCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);
  TILING_DATA_FIELD_DEF(uint32_t, outFeatures);
  TILING_DATA_FIELD_DEF(uint32_t, poolKernelSize);
  TILING_DATA_FIELD_DEF(uint32_t, pooledSize);
  TILING_DATA_FIELD_DEF(float, scaleFactor);
  TILING_DATA_FIELD_DEF(float, invPoolSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MatmulAvgPoolGeluScaleMaxCustom, MatmulAvgPoolGeluScaleMaxCustomTilingData)
}
