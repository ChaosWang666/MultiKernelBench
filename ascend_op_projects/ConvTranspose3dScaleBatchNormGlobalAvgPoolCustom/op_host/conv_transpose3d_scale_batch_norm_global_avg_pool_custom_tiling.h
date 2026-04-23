
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ConvTranspose3dScaleBatchNormGlobalAvgPoolCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, numGroups);
  TILING_DATA_FIELD_DEF(uint32_t, groupSize);
  TILING_DATA_FIELD_DEF(uint32_t, tileLength);
  TILING_DATA_FIELD_DEF(uint32_t, groupsPerBlock);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ConvTranspose3dScaleBatchNormGlobalAvgPoolCustom, ConvTranspose3dScaleBatchNormGlobalAvgPoolCustomTilingData)
}
