
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ConvTranspose3dMaxPoolSoftmaxSubtractSwishMaxCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalRows);
  TILING_DATA_FIELD_DEF(uint32_t, rowsPerCore);
  TILING_DATA_FIELD_DEF(uint32_t, tileRows);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ConvTranspose3dMaxPoolSoftmaxSubtractSwishMaxCustom, ConvTranspose3dMaxPoolSoftmaxSubtractSwishMaxCustomTilingData)
}
