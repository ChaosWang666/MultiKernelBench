
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GridSampleRandomWarpCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);
  TILING_DATA_FIELD_DEF(uint32_t, channels);
  TILING_DATA_FIELD_DEF(uint32_t, inH);
  TILING_DATA_FIELD_DEF(uint32_t, inW);
  TILING_DATA_FIELD_DEF(uint32_t, outH);
  TILING_DATA_FIELD_DEF(uint32_t, outW);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GridSampleRandomWarpCustom, GridSampleRandomWarpCustomTilingData)
}
