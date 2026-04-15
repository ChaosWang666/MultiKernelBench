
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ConvTransposed3dSquareInputSquareKernelCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, size);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ConvTransposed3dSquareInputSquareKernelCustom, ConvTransposed3dSquareInputSquareKernelCustomTilingData)
}
