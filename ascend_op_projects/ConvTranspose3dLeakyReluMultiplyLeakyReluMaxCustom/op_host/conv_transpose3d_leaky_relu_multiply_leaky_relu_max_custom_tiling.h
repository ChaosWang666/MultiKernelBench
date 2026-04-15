
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ConvTranspose3dLeakyReluMultiplyLeakyReluMaxCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);
  TILING_DATA_FIELD_DEF(uint32_t, channels);
  TILING_DATA_FIELD_DEF(uint32_t, depthIn);
  TILING_DATA_FIELD_DEF(uint32_t, heightIn);
  TILING_DATA_FIELD_DEF(uint32_t, widthIn);
  TILING_DATA_FIELD_DEF(uint32_t, depthOut);
  TILING_DATA_FIELD_DEF(uint32_t, heightOut);
  TILING_DATA_FIELD_DEF(uint32_t, widthOut);
  TILING_DATA_FIELD_DEF(uint32_t, totalOutputElems);
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ConvTranspose3dLeakyReluMultiplyLeakyReluMaxCustom, ConvTranspose3dLeakyReluMultiplyLeakyReluMaxCustomTilingData)
}
