
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MultiQueryAttentionCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);
  TILING_DATA_FIELD_DEF(uint32_t, seqLen);
  TILING_DATA_FIELD_DEF(uint32_t, numHeads);
  TILING_DATA_FIELD_DEF(uint32_t, headDim);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MultiQueryAttentionCustom, MultiQueryAttentionCustomTilingData)
}
