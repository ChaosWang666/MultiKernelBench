
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SquareMatrixMultiplicationCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, N);
  TILING_DATA_FIELD_DEF(uint32_t, tileM);
  TILING_DATA_FIELD_DEF(uint32_t, tileN);
  TILING_DATA_FIELD_DEF(uint32_t, tileK);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SquareMatrixMultiplicationCustom, SquareMatrixMultiplicationCustomTilingData)
}
