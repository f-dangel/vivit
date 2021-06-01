from backpack.core.derivatives.avgpool1d import AvgPool1DDerivatives
from backpack.core.derivatives.avgpool2d import AvgPool2DDerivatives
from backpack.core.derivatives.avgpool3d import AvgPool3DDerivatives
from backpack.core.derivatives.maxpool1d import MaxPool1DDerivatives
from backpack.core.derivatives.maxpool2d import MaxPool2DDerivatives
from backpack.core.derivatives.maxpool3d import MaxPool3DDerivatives

from vivit.extensions.secondorder.sqrt_ggn.sqrt_ggn_base import SqrtGGNBaseModule


class SqrtGGNMaxPool1d(SqrtGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=MaxPool1DDerivatives())


class SqrtGGNMaxPool2d(SqrtGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=MaxPool2DDerivatives())


class SqrtGGNAvgPool1d(SqrtGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=AvgPool1DDerivatives())


class SqrtGGNMaxPool3d(SqrtGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=MaxPool3DDerivatives())


class SqrtGGNAvgPool2d(SqrtGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=AvgPool2DDerivatives())


class SqrtGGNAvgPool3d(SqrtGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=AvgPool3DDerivatives())
