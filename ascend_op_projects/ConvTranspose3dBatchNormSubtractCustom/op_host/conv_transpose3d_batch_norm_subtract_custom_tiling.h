
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ConvTranspose3dBatchNormSubtractCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalRows);
  TILING_DATA_FIELD_DEF(uint32_t, rowLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileLength);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ConvTranspose3dBatchNormSubtractCustom, ConvTranspose3dBatchNormSubtractCustomTilingData)
}
