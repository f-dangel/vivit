from backpack.core.derivatives.conv1d import Conv1DDerivatives

from vivit.extensions.secondorder.sqrt_ggn.convnd import SqrtGGNConvND


class SqrtGGNConv1d(SqrtGGNConvND):
    def __init__(self):
        super().__init__(
            derivatives=Conv1DDerivatives(),
            N=1,
            params=["bias", "weight"],
        )
