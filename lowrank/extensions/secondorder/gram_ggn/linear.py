from backpack.core.derivatives.linear import LinearDerivatives
from lowrank.extensions.secondorder.gram_ggn.gram_ggn_base import GramGGNBaseModule


class GramGGNLinear(GramGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=LinearDerivatives(), params=["bias", "weight"])
