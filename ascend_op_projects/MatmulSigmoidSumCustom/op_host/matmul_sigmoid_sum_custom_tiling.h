
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MatmulSigmoidSumCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);
  TILING_DATA_FIELD_DEF(uint32_t, hiddenSize);
  TILING_DATA_FIELD_DEF(uint32_t, tileLength);
  TILING_DATA_FIELD_DEF(uint32_t, numTiles);
  TILING_DATA_FIELD_DEF(uint32_t, rowsPerCore);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MatmulSigmoidSumCustom, MatmulSigmoidSumCustomTilingData)
}
