
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MatrixVectorMultiplicationCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, M);
  TILING_DATA_FIELD_DEF(uint32_t, K);
  TILING_DATA_FIELD_DEF(uint32_t, tileK);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(MatrixVectorMultiplicationCustom, MatrixVectorMultiplicationCustomTilingData)
}
