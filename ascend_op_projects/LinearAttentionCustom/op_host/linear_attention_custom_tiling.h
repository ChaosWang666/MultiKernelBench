
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(LinearAttentionCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);
  TILING_DATA_FIELD_DEF(uint32_t, seqLen);
  TILING_DATA_FIELD_DEF(uint32_t, dModel);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LinearAttentionCustom, LinearAttentionCustomTilingData)
}
