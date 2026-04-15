
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(BicubicUpsampleCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);
  TILING_DATA_FIELD_DEF(uint32_t, channels);
  TILING_DATA_FIELD_DEF(uint32_t, inputHeight);
  TILING_DATA_FIELD_DEF(uint32_t, inputWidth);
  TILING_DATA_FIELD_DEF(uint32_t, outputHeight);
  TILING_DATA_FIELD_DEF(uint32_t, outputWidth);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(BicubicUpsampleCustom, BicubicUpsampleCustomTilingData)
}
