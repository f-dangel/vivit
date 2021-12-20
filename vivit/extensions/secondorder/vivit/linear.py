"""Contains extension for the linear layer used by ``ViViTGGN{Exact, MC}``."""
from backpack.core.derivatives.linear import LinearDerivatives

from vivit.extensions.secondorder.vivit.base import ViViTGGNBaseModule


class ViViTGGNLinear(ViViTGGNBaseModule):
    """``ViViTGGN{Exact, MC}`` extension for ``torch.nn.Linear`` module."""

    def __init__(self):
        """Pass derivatives for ``torch.nn.Linear`` module."""
        super().__init__(LinearDerivatives(), params=["bias", "weight"])
