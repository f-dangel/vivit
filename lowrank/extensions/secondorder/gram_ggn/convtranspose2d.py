from backpack.core.derivatives.conv_transpose2d import ConvTranspose2DDerivatives
from lowrank.extensions.secondorder.gram_ggn.convtransposend import (
    GramGGNConvTransposeND,
)


class GramGGNConvTranspose2d(GramGGNConvTransposeND):
    def __init__(self):
        super().__init__(
            derivatives=ConvTranspose2DDerivatives(),
            N=2,
            params=["bias", "weight"],
        )
