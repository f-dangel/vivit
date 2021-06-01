from backpack.core.derivatives.elu import ELUDerivatives
from backpack.core.derivatives.leakyrelu import LeakyReLUDerivatives
from backpack.core.derivatives.logsigmoid import LogSigmoidDerivatives
from backpack.core.derivatives.relu import ReLUDerivatives
from backpack.core.derivatives.selu import SELUDerivatives
from backpack.core.derivatives.sigmoid import SigmoidDerivatives
from backpack.core.derivatives.tanh import TanhDerivatives

from vivit.extensions.secondorder.sqrt_ggn.sqrt_ggn_base import SqrtGGNBaseModule


class SqrtGGNReLU(SqrtGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=ReLUDerivatives())


class SqrtGGNSigmoid(SqrtGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=SigmoidDerivatives())


class SqrtGGNTanh(SqrtGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=TanhDerivatives())


class SqrtGGNELU(SqrtGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=ELUDerivatives())


class SqrtGGNSELU(SqrtGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=SELUDerivatives())


class SqrtGGNLeakyReLU(SqrtGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=LeakyReLUDerivatives())


class SqrtGGNLogSigmoid(SqrtGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=LogSigmoidDerivatives())
