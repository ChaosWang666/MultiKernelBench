
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(EmbeddingCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, numIndices);
  TILING_DATA_FIELD_DEF(uint32_t, embeddingDim);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(EmbeddingCustom, EmbeddingCustomTilingData)
}
