from backpack.core.derivatives.conv_transpose1d import \
    ConvTranspose1DDerivatives
from lowrank.extensions.secondorder.gram_ggn.convtransposend import \
    GramGGNConvTransposeND


class GramGGNConvTranspose1d(GramGGNConvTransposeND):
    def __init__(self):
        super().__init__(
            derivatives=ConvTranspose1DDerivatives(),
            N=1,
            params=["bias", "weight"],
        )
