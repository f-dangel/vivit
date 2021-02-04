from backpack.core.derivatives.dropout import DropoutDerivatives
from lowrank.extensions.secondorder.gram_ggn.gram_ggn_base import \
    GramGGNBaseModule


class GramGGNDropout(GramGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=DropoutDerivatives())
