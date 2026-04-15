
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MaxPooling3dCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);
  TILING_DATA_FIELD_DEF(uint32_t, channels);
  TILING_DATA_FIELD_DEF(uint32_t, dim1);
  TILING_DATA_FIELD_DEF(uint32_t, dim2);
  TILING_DATA_FIELD_DEF(uint32_t, dim3);
  TILING_DATA_FIELD_DEF(uint32_t, outDim1);
  TILING_DATA_FIELD_DEF(uint32_t, outDim2);
  TILING_DATA_FIELD_DEF(uint32_t, outDim3);
  TILING_DATA_FIELD_DEF(uint32_t, kernelSize);
  TILING_DATA_FIELD_DEF(uint32_t, stride);
  TILING_DATA_FIELD_DEF(uint32_t, padding);
  TILING_DATA_FIELD_DEF(uint32_t, totalOutputElements);
  TILING_DATA_FIELD_DEF(uint32_t, blockDim);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MaxPooling3dCustom, MaxPooling3dCustomTilingData)
}
