
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(CrossEntropyLossCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);
  TILING_DATA_FIELD_DEF(uint32_t, numClasses);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(CrossEntropyLossCustom, CrossEntropyLossCustomTilingData)
}
