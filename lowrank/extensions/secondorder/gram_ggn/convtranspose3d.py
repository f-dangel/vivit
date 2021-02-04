from backpack.core.derivatives.conv_transpose3d import \
    ConvTranspose3DDerivatives
from lowrank.extensions.secondorder.gram_ggn.convtransposend import \
    GramGGNConvTransposeND


class GramGGNConvTranspose3d(GramGGNConvTransposeND):
    def __init__(self):
        super().__init__(
            derivatives=ConvTranspose3DDerivatives(),
            N=3,
            params=["bias", "weight"],
        )
