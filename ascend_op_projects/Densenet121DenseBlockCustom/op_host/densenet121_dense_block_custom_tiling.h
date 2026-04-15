
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(Densenet121DenseBlockCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, numLayers);
  TILING_DATA_FIELD_DEF(uint32_t, numInputFeatures);
  TILING_DATA_FIELD_DEF(uint32_t, growthRate);
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);
  TILING_DATA_FIELD_DEF(uint32_t, height);
  TILING_DATA_FIELD_DEF(uint32_t, width);
  TILING_DATA_FIELD_DEF(uint32_t, totalElements);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Densenet121DenseBlockCustom, Densenet121DenseBlockCustomTilingData)
}
