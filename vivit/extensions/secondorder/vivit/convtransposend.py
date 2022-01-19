"""Contains transpose convolution layer extensions used by ``ViViTGGN{Exact, MC}``."""
from backpack.core.derivatives.conv_transpose1d import ConvTranspose1DDerivatives
from backpack.core.derivatives.conv_transpose2d import ConvTranspose2DDerivatives
from backpack.core.derivatives.conv_transpose3d import ConvTranspose3DDerivatives

from vivit.extensions.secondorder.vivit.base import ViViTGGNBaseModule


class ViViTGGNConvTranspose1d(ViViTGGNBaseModule):
    """``ViViTGGN{Exact, MC}`` extension for ``torch.nn.ConvTranspose1d`` module."""

    def __init__(self):
        """Pass derivatives for ``torch.nn.ConvTranspose1d`` module."""
        super().__init__(ConvTranspose1DDerivatives(), params=["bias", "weight"])


class ViViTGGNConvTranspose2d(ViViTGGNBaseModule):
    """``ViViTGGN{Exact, MC}`` extension for ``torch.nn.ConvTranspose2d`` module."""

    def __init__(self):
        """Pass derivatives for ``torch.nn.ConvTranspose2d`` module."""
        super().__init__(ConvTranspose2DDerivatives(), params=["bias", "weight"])


class ViViTGGNConvTranspose3d(ViViTGGNBaseModule):
    """``ViViTGGN{Exact, MC}`` extension for ``torch.nn.ConvTranspose3d`` module."""

    def __init__(self):
        """Pass derivatives for ``torch.nn.ConvTranspose3d`` module."""
        super().__init__(ConvTranspose3DDerivatives(), params=["bias", "weight"])
