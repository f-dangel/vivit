from backpack.core.derivatives.conv_transpose2d import ConvTranspose2DDerivatives

from vivit.extensions.secondorder.sqrt_ggn.convtransposend import SqrtGGNConvTransposeND


class SqrtGGNConvTranspose2d(SqrtGGNConvTransposeND):
    def __init__(self):
        super().__init__(
            derivatives=ConvTranspose2DDerivatives(),
            N=2,
            params=["bias", "weight"],
        )
