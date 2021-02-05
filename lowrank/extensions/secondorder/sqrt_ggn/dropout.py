from backpack.core.derivatives.dropout import DropoutDerivatives
from lowrank.extensions.secondorder.sqrt_ggn.sqrt_ggn_base import SqrtGGNBaseModule


class SqrtGGNDropout(SqrtGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=DropoutDerivatives())
