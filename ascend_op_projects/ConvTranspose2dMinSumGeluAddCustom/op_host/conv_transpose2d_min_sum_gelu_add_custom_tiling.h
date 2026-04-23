
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ConvTranspose2dMinSumGeluAddCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalN);
  TILING_DATA_FIELD_DEF(uint32_t, totalC);
  TILING_DATA_FIELD_DEF(uint32_t, totalH);
  TILING_DATA_FIELD_DEF(uint32_t, totalW);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ConvTranspose2dMinSumGeluAddCustom, ConvTranspose2dMinSumGeluAddCustomTilingData)
}
