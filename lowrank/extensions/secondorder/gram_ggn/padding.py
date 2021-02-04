from backpack.core.derivatives.zeropad2d import ZeroPad2dDerivatives
from lowrank.extensions.secondorder.gram_ggn.gram_ggn_base import GramGGNBaseModule


class GramGGNZeroPad2d(GramGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=ZeroPad2dDerivatives())
