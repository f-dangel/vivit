"""Contains extensions for batch normalization used by ``ViViTGGN{Exact, MC}``."""

from backpack.core.derivatives.batchnorm_nd import BatchNormNdDerivatives

from vivit.extensions.secondorder.vivit.base import ViViTGGNBaseModule


class ViViTGGNBatchNormNd(ViViTGGNBaseModule):
    """``ViViTGGN{Exact, MC}`` extension for ``torch.nn.BatchNormNd`` module."""

    def __init__(self):
        """Pass derivatives for ``torch.nn.Conv1d`` module."""
        super().__init__(BatchNormNdDerivatives(), params=["bias", "weight"])
