
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(Conv2dAvgPoolSigmoidSumCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);
  TILING_DATA_FIELD_DEF(uint32_t, batchElements);
  TILING_DATA_FIELD_DEF(uint32_t, tileLength);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Conv2dAvgPoolSigmoidSumCustom, Conv2dAvgPoolSigmoidSumCustomTilingData)
}
