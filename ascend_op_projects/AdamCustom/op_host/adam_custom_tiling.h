
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AdamCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);
  TILING_DATA_FIELD_DEF(float, beta1);
  TILING_DATA_FIELD_DEF(float, beta2);
  TILING_DATA_FIELD_DEF(float, lr);
  TILING_DATA_FIELD_DEF(float, eps);
  TILING_DATA_FIELD_DEF(float, beta1CorrInv);
  TILING_DATA_FIELD_DEF(float, beta2CorrInv);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AdamCustom, AdamCustomTilingData)
}
