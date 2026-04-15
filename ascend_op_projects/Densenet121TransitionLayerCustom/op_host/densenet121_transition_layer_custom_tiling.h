
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(Densenet121TransitionLayerCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, batch);
  TILING_DATA_FIELD_DEF(uint32_t, height);
  TILING_DATA_FIELD_DEF(uint32_t, width);
  TILING_DATA_FIELD_DEF(uint32_t, numInputFeatures);
  TILING_DATA_FIELD_DEF(uint32_t, numOutputFeatures);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Densenet121TransitionLayerCustom, Densenet121TransitionLayerCustomTilingData)
}
