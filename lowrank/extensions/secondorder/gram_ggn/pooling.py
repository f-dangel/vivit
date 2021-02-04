from backpack.core.derivatives.avgpool1d import AvgPool1DDerivatives
from backpack.core.derivatives.avgpool2d import AvgPool2DDerivatives
from backpack.core.derivatives.avgpool3d import AvgPool3DDerivatives
from backpack.core.derivatives.maxpool1d import MaxPool1DDerivatives
from backpack.core.derivatives.maxpool2d import MaxPool2DDerivatives
from backpack.core.derivatives.maxpool3d import MaxPool3DDerivatives
from lowrank.extensions.secondorder.gram_ggn.gram_ggn_base import GramGGNBaseModule


class GramGGNMaxPool1d(GramGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=MaxPool1DDerivatives())


class GramGGNMaxPool2d(GramGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=MaxPool2DDerivatives())


class GramGGNAvgPool1d(GramGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=AvgPool1DDerivatives())


class GramGGNMaxPool3d(GramGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=MaxPool3DDerivatives())


class GramGGNAvgPool2d(GramGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=AvgPool2DDerivatives())


class GramGGNAvgPool3d(GramGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=AvgPool3DDerivatives())
