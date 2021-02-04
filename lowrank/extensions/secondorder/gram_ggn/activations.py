from backpack.core.derivatives.elu import ELUDerivatives
from backpack.core.derivatives.leakyrelu import LeakyReLUDerivatives
from backpack.core.derivatives.logsigmoid import LogSigmoidDerivatives
from backpack.core.derivatives.relu import ReLUDerivatives
from backpack.core.derivatives.selu import SELUDerivatives
from backpack.core.derivatives.sigmoid import SigmoidDerivatives
from backpack.core.derivatives.tanh import TanhDerivatives
from lowrank.extensions.secondorder.gram_ggn.gram_ggn_base import \
    GramGGNBaseModule


class GramGGNReLU(GramGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=ReLUDerivatives())


class GramGGNSigmoid(GramGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=SigmoidDerivatives())


class GramGGNTanh(GramGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=TanhDerivatives())


class GramGGNELU(GramGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=ELUDerivatives())


class GramGGNSELU(GramGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=SELUDerivatives())


class GramGGNLeakyReLU(GramGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=LeakyReLUDerivatives())


class GramGGNLogSigmoid(GramGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=LogSigmoidDerivatives())
