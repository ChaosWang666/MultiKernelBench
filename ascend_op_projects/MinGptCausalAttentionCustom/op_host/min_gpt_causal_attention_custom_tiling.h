
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MinGptCausalAttentionCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batch);
  TILING_DATA_FIELD_DEF(uint32_t, seqLen);
  TILING_DATA_FIELD_DEF(uint32_t, nEmb);
  TILING_DATA_FIELD_DEF(uint32_t, nHead);
  TILING_DATA_FIELD_DEF(uint32_t, maxSeqlen);
  TILING_DATA_FIELD_DEF(float, attnPdrop);
  TILING_DATA_FIELD_DEF(float, residPdrop);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MinGptCausalAttentionCustom, MinGptCausalAttentionCustomTilingData)
}
