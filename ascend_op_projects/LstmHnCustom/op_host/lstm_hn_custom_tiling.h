
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(LstmHnCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);
  TILING_DATA_FIELD_DEF(uint32_t, seqLength);
  TILING_DATA_FIELD_DEF(uint32_t, inputSize);
  TILING_DATA_FIELD_DEF(uint32_t, hiddenSize);
  TILING_DATA_FIELD_DEF(uint32_t, numLayers);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LstmHnCustom, LstmHnCustomTilingData)
}
