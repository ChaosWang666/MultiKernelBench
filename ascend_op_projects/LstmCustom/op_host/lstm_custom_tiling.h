
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(LstmCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);
  TILING_DATA_FIELD_DEF(uint32_t, seqLength);
  TILING_DATA_FIELD_DEF(uint32_t, inputSize);
  TILING_DATA_FIELD_DEF(uint32_t, hiddenSize);
  TILING_DATA_FIELD_DEF(uint32_t, numLayers);
  TILING_DATA_FIELD_DEF(uint32_t, tileSeqLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileHiddenSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LstmCustom, LstmCustomTilingData)
}
