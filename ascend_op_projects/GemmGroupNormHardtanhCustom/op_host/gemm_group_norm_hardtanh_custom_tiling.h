
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GemmGroupNormHardtanhCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalRows);
  TILING_DATA_FIELD_DEF(uint32_t, outFeatures);
  TILING_DATA_FIELD_DEF(uint32_t, numGroups);
  TILING_DATA_FIELD_DEF(uint32_t, rowsPerBlock);
  TILING_DATA_FIELD_DEF(float, htMin);
  TILING_DATA_FIELD_DEF(float, htMax);
  TILING_DATA_FIELD_DEF(float, eps);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GemmGroupNormHardtanhCustom, GemmGroupNormHardtanhCustomTilingData)
}
