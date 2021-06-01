from backpack.core.derivatives.conv_transpose3d import ConvTranspose3DDerivatives

from vivit.extensions.secondorder.sqrt_ggn.convtransposend import SqrtGGNConvTransposeND


class SqrtGGNConvTranspose3d(SqrtGGNConvTransposeND):
    def __init__(self):
        super().__init__(
            derivatives=ConvTranspose3DDerivatives(),
            N=3,
            params=["bias", "weight"],
        )
