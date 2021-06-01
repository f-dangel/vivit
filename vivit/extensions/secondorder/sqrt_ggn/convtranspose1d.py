from backpack.core.derivatives.conv_transpose1d import ConvTranspose1DDerivatives

from vivit.extensions.secondorder.sqrt_ggn.convtransposend import SqrtGGNConvTransposeND


class SqrtGGNConvTranspose1d(SqrtGGNConvTransposeND):
    def __init__(self):
        super().__init__(
            derivatives=ConvTranspose1DDerivatives(),
            N=1,
            params=["bias", "weight"],
        )
