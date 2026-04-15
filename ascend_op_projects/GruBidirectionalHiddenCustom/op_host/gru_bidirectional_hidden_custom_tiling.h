
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GruBidirectionalHiddenCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, seqLen);
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);
  TILING_DATA_FIELD_DEF(uint32_t, inputSize);
  TILING_DATA_FIELD_DEF(uint32_t, hiddenSize);
  TILING_DATA_FIELD_DEF(uint32_t, numLayers);
  TILING_DATA_FIELD_DEF(uint32_t, tileSeqLen);
  TILING_DATA_FIELD_DEF(uint32_t, tileBatchSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GruBidirectionalHiddenCustom, GruBidirectionalHiddenCustomTilingData)
}
