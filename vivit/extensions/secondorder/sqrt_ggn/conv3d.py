from backpack.core.derivatives.conv3d import Conv3DDerivatives

from vivit.extensions.secondorder.sqrt_ggn.convnd import SqrtGGNConvND


class SqrtGGNConv3d(SqrtGGNConvND):
    def __init__(self):
        super().__init__(
            derivatives=Conv3DDerivatives(),
            N=3,
            params=["bias", "weight"],
        )
