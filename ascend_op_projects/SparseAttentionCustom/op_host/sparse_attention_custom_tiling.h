
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SparseAttentionCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);
  TILING_DATA_FIELD_DEF(uint32_t, numHeads);
  TILING_DATA_FIELD_DEF(uint32_t, seqLen);
  TILING_DATA_FIELD_DEF(uint32_t, headDim);
  TILING_DATA_FIELD_DEF(uint32_t, windowSize);
  TILING_DATA_FIELD_DEF(uint32_t, totalBatchHeads);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SparseAttentionCustom, SparseAttentionCustomTilingData)
}
