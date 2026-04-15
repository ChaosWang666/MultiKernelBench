
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GemmSubtractGlobalAvgPoolLogSumExpGeluResidualAddCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);
  TILING_DATA_FIELD_DEF(uint32_t, outFeatures);
  TILING_DATA_FIELD_DEF(uint32_t, inFeatures);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GemmSubtractGlobalAvgPoolLogSumExpGeluResidualAddCustom, GemmSubtractGlobalAvgPoolLogSumExpGeluResidualAddCustomTilingData)
}
