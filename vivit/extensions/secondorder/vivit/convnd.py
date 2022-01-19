"""Contains extensions for convolution layers used by ``ViViTGGN{Exact, MC}``."""
from backpack.core.derivatives.conv1d import Conv1DDerivatives
from backpack.core.derivatives.conv2d import Conv2DDerivatives
from backpack.core.derivatives.conv3d import Conv3DDerivatives

from vivit.extensions.secondorder.vivit.base import ViViTGGNBaseModule


class ViViTGGNConv1d(ViViTGGNBaseModule):
    """``ViViTGGN{Exact, MC}`` extension for ``torch.nn.Conv1d`` module."""

    def __init__(self):
        """Pass derivatives for ``torch.nn.Conv1d`` module."""
        super().__init__(Conv1DDerivatives(), params=["bias", "weight"])


class ViViTGGNConv2d(ViViTGGNBaseModule):
    """``ViViTGGN{Exact, MC}`` extension for ``torch.nn.Conv2d`` module."""

    def __init__(self):
        """Pass derivatives for ``torch.nn.Conv2d`` module."""
        super().__init__(Conv2DDerivatives(), params=["bias", "weight"])


class ViViTGGNConv3d(ViViTGGNBaseModule):
    """``ViViTGGN{Exact, MC}`` extension for ``torch.nn.Conv3d`` module."""

    def __init__(self):
        """Pass derivatives for ``torch.nn.Conv3d`` module."""
        super().__init__(Conv3DDerivatives(), params=["bias", "weight"])
