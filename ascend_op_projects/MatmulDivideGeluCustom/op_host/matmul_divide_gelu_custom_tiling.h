
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MatmulDivideGeluCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);
  TILING_DATA_FIELD_DEF(uint32_t, inputSize);
  TILING_DATA_FIELD_DEF(uint32_t, outputSize);
  TILING_DATA_FIELD_DEF(float, divisor);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MatmulDivideGeluCustom, MatmulDivideGeluCustomTilingData)
}
