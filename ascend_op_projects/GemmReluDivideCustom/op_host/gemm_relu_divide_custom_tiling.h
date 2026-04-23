
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(GemmReluDivideCustomTilingData)
  TILING_DATA_FIELD_DEF(float, divisor);
  TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cubeTilingData);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GemmReluDivideCustom, GemmReluDivideCustomTilingData)
}
