
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(BilinearUpsampleCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);
  TILING_DATA_FIELD_DEF(uint32_t, channels);
  TILING_DATA_FIELD_DEF(uint32_t, inputH);
  TILING_DATA_FIELD_DEF(uint32_t, inputW);
  TILING_DATA_FIELD_DEF(uint32_t, outputH);
  TILING_DATA_FIELD_DEF(uint32_t, outputW);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BilinearUpsampleCustom, BilinearUpsampleCustomTilingData)
}
