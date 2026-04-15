
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(KvCachedSpeculativeAttentionCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);
  TILING_DATA_FIELD_DEF(uint32_t, qLen);
  TILING_DATA_FIELD_DEF(uint32_t, kvLen);
  TILING_DATA_FIELD_DEF(uint32_t, dModel);
  TILING_DATA_FIELD_DEF(float, scale);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(KvCachedSpeculativeAttentionCustom, KvCachedSpeculativeAttentionCustomTilingData)
}
