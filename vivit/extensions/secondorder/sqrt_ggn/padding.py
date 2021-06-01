from backpack.core.derivatives.zeropad2d import ZeroPad2dDerivatives

from vivit.extensions.secondorder.sqrt_ggn.sqrt_ggn_base import SqrtGGNBaseModule


class SqrtGGNZeroPad2d(SqrtGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=ZeroPad2dDerivatives())
