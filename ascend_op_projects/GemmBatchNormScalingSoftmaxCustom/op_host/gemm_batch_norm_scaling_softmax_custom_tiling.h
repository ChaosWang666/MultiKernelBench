
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GemmBatchNormScalingSoftmaxCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);
  TILING_DATA_FIELD_DEF(uint32_t, inFeatures);
  TILING_DATA_FIELD_DEF(uint32_t, outFeatures);
  TILING_DATA_FIELD_DEF(float, bnEps);
  TILING_DATA_FIELD_DEF(float, bnMomentum);
  TILING_DATA_FIELD_DEF(uint32_t, scaleShape0);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GemmBatchNormScalingSoftmaxCustom, GemmBatchNormScalingSoftmaxCustomTilingData)
}
