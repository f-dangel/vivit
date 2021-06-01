from vivit.extensions.secondorder.sqrt_ggn.sqrt_ggn_base import SqrtGGNBaseModule


class SqrtGGNConvTransposeND(SqrtGGNBaseModule):
    def __init__(self, derivatives, N, params=None):
        super().__init__(derivatives=derivatives, params=params)
        self.N = N
