
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MiniGptBlockCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);
  TILING_DATA_FIELD_DEF(uint32_t, seqLen);
  TILING_DATA_FIELD_DEF(uint32_t, embdDim);
  TILING_DATA_FIELD_DEF(uint32_t, headNum);
  TILING_DATA_FIELD_DEF(uint32_t, headSize);
  TILING_DATA_FIELD_DEF(uint32_t, mlpHiddenDim);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MiniGptBlockCustom, MiniGptBlockCustomTilingData)
}
