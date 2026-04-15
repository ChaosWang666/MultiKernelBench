
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ScaledDotProductAttentionLongContextCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);
  TILING_DATA_FIELD_DEF(uint32_t, seqLen);
  TILING_DATA_FIELD_DEF(uint32_t, dModel);
  TILING_DATA_FIELD_DEF(uint32_t, blockQ);
  TILING_DATA_FIELD_DEF(uint32_t, blockKV);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ScaledDotProductAttentionLongContextCustom, ScaledDotProductAttentionLongContextCustomTilingData)
}
