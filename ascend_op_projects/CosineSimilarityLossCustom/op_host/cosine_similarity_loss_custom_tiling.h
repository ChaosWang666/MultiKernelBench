
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(CosineSimilarityLossCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);
  TILING_DATA_FIELD_DEF(uint32_t, featureSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(CosineSimilarityLossCustom, CosineSimilarityLossCustomTilingData)
}
