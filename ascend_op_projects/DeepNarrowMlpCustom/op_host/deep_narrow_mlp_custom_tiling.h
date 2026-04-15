
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(DeepNarrowMlpCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, inputSize);
  TILING_DATA_FIELD_DEF(uint32_t, hiddenLayerSizes);
  TILING_DATA_FIELD_DEF(uint32_t, outputSize);
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DeepNarrowMlpCustom, DeepNarrowMlpCustomTilingData)
}
