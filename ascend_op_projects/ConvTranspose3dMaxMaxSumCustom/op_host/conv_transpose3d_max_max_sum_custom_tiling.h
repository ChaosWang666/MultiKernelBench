
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ConvTranspose3dMaxMaxSumCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, B);
  TILING_DATA_FIELD_DEF(uint32_t, C);
  TILING_DATA_FIELD_DEF(uint32_t, DHW);
  TILING_DATA_FIELD_DEF(uint32_t, tileLen);
  TILING_DATA_FIELD_DEF(uint32_t, tilesPerBatch);
  TILING_DATA_FIELD_DEF(uint32_t, totalTiles);
  TILING_DATA_FIELD_DEF(uint32_t, tilesPerCore);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ConvTranspose3dMaxMaxSumCustom, ConvTranspose3dMaxMaxSumCustomTilingData)
}
