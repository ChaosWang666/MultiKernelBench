
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GemmSubtractGlobalAvgPoolLogSumExpGeluResidualAddCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);
  TILING_DATA_FIELD_DEF(uint32_t, nLen);
  TILING_DATA_FIELD_DEF(uint32_t, mLen);
  TILING_DATA_FIELD_DEF(uint32_t, rowsPerBlock);
  TILING_DATA_FIELD_DEF(uint32_t, tileSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GemmSubtractGlobalAvgPoolLogSumExpGeluResidualAddCustom, GemmSubtractGlobalAvgPoolLogSumExpGeluResidualAddCustomTilingData)
}
