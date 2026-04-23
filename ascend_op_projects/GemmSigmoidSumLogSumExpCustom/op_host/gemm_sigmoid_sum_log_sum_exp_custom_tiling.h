
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GemmSigmoidSumLogSumExpCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);
  TILING_DATA_FIELD_DEF(uint32_t, featureSize);
  TILING_DATA_FIELD_DEF(uint32_t, coreNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GemmSigmoidSumLogSumExpCustom, GemmSigmoidSumLogSumExpCustomTilingData)
}
