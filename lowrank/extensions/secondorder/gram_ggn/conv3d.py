from backpack.core.derivatives.conv3d import Conv3DDerivatives
from lowrank.extensions.secondorder.gram_ggn.convnd import GramGGNConvND


class GramGGNConv3d(GramGGNConvND):
    def __init__(self):
        super().__init__(
            derivatives=Conv3DDerivatives(),
            N=3,
            params=["bias", "weight"],
        )
