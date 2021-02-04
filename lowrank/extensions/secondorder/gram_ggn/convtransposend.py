from lowrank.extensions.secondorder.gram_ggn.gram_ggn_base import \
    GramGGNBaseModule


class GramGGNConvTransposeND(GramGGNBaseModule):
    def __init__(self, derivatives, N, params=None):
        super().__init__(derivatives=derivatives, params=params)
        self.N = N
