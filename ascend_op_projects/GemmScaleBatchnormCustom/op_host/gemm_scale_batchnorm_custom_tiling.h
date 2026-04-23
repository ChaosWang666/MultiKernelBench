
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GemmScaleBatchnormCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);
  TILING_DATA_FIELD_DEF(uint32_t, features);
  TILING_DATA_FIELD_DEF(uint32_t, rowsPerCore);
  TILING_DATA_FIELD_DEF(uint32_t, tileRows);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GemmScaleBatchnormCustom, GemmScaleBatchnormCustomTilingData)
}
