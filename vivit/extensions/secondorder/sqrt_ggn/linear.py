from backpack.core.derivatives.linear import LinearDerivatives

from vivit.extensions.secondorder.sqrt_ggn.sqrt_ggn_base import SqrtGGNBaseModule


class SqrtGGNLinear(SqrtGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=LinearDerivatives(), params=["bias", "weight"])
