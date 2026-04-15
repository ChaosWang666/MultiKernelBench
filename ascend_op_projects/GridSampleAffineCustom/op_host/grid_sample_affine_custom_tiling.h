
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GridSampleAffineCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);
  TILING_DATA_FIELD_DEF(uint32_t, channels);
  TILING_DATA_FIELD_DEF(uint32_t, inH);
  TILING_DATA_FIELD_DEF(uint32_t, inW);
  TILING_DATA_FIELD_DEF(uint32_t, outH);
  TILING_DATA_FIELD_DEF(uint32_t, outW);
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GridSampleAffineCustom, GridSampleAffineCustomTilingData)
}
