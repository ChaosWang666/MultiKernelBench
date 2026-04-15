
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(Conv3dDivideMaxGlobalAvgPoolBiasAddSumCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);
  TILING_DATA_FIELD_DEF(uint32_t, inChannels);
  TILING_DATA_FIELD_DEF(uint32_t, outChannels);
  TILING_DATA_FIELD_DEF(uint32_t, depth);
  TILING_DATA_FIELD_DEF(uint32_t, height);
  TILING_DATA_FIELD_DEF(uint32_t, width);
  TILING_DATA_FIELD_DEF(uint32_t, kernelDepth);
  TILING_DATA_FIELD_DEF(uint32_t, kernelHeight);
  TILING_DATA_FIELD_DEF(uint32_t, kernelWidth);
  TILING_DATA_FIELD_DEF(float, divisor);
  TILING_DATA_FIELD_DEF(uint32_t, poolDepth);
  TILING_DATA_FIELD_DEF(uint32_t, poolHeight);
  TILING_DATA_FIELD_DEF(uint32_t, poolWidth);
  TILING_DATA_FIELD_DEF(uint32_t, sumDim);
  TILING_DATA_FIELD_DEF(uint32_t, totalElements);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Conv3dDivideMaxGlobalAvgPoolBiasAddSumCustom, Conv3dDivideMaxGlobalAvgPoolBiasAddSumCustomTilingData)
}
