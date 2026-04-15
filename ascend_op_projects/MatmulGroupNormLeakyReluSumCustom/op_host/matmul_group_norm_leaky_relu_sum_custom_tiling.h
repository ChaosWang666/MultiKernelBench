
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MatmulGroupNormLeakyReluSumCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);
  TILING_DATA_FIELD_DEF(uint32_t, inputSize);
  TILING_DATA_FIELD_DEF(uint32_t, hiddenSize);
  TILING_DATA_FIELD_DEF(uint32_t, numGroups);
  TILING_DATA_FIELD_DEF(float, eps);
  TILING_DATA_FIELD_DEF(float, negativeSlope);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MatmulGroupNormLeakyReluSumCustom, MatmulGroupNormLeakyReluSumCustomTilingData)
}
