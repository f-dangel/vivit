from backpack.core.derivatives.conv2d import Conv2DDerivatives

from vivit.extensions.secondorder.sqrt_ggn.convnd import SqrtGGNConvND


class SqrtGGNConv2d(SqrtGGNConvND):
    def __init__(self):
        super().__init__(
            derivatives=Conv2DDerivatives(),
            N=2,
            params=["bias", "weight"],
        )
