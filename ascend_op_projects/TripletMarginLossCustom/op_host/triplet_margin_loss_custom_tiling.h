
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(TripletMarginLossCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batchSize);
  TILING_DATA_FIELD_DEF(uint32_t, dimSize);
  TILING_DATA_FIELD_DEF(float, margin);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(TripletMarginLossCustom, TripletMarginLossCustomTilingData)
}
