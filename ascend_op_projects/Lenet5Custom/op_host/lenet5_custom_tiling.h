
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(Lenet5CustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);
  TILING_DATA_FIELD_DEF(uint32_t, channel);
  TILING_DATA_FIELD_DEF(uint32_t, height);
  TILING_DATA_FIELD_DEF(uint32_t, width);
  TILING_DATA_FIELD_DEF(uint32_t, kernelH);
  TILING_DATA_FIELD_DEF(uint32_t, kernelW);
  TILING_DATA_FIELD_DEF(uint32_t, strideH);
  TILING_DATA_FIELD_DEF(uint32_t, strideW);
  TILING_DATA_FIELD_DEF(uint32_t, padH);
  TILING_DATA_FIELD_DEF(uint32_t, padW);
  TILING_DATA_FIELD_DEF(uint32_t, outChannel);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Lenet5Custom, Lenet5CustomTilingData)
}
