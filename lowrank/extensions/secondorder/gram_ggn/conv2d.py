from backpack.core.derivatives.conv2d import Conv2DDerivatives
from lowrank.extensions.secondorder.gram_ggn.convnd import GramGGNConvND


class GramGGNConv2d(GramGGNConvND):
    def __init__(self):
        super().__init__(
            derivatives=Conv2DDerivatives(),
            N=2,
            params=["bias", "weight"],
        )
