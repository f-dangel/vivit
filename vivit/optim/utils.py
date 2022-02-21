"""Utility functions for ``vivit.optim``."""

from typing import List, Union

from backpack.extensions import SqrtGGNExact, SqrtGGNMC


def get_sqrt_ggn_extension(
    subsampling: Union[None, List[int]], mc_samples: int
) -> Union[SqrtGGNExact, SqrtGGNMC]:
    """Instantiate ``SqrtGGN{Exact, MC} extension.

    Args:
        subsampling: Indices of active samples.
        mc_samples: Number of MC-samples to approximate the loss Hessian. ``0``
            uses the exact loss Hessian.

    Returns:
        Instantiated SqrtGGN extension.
    """
    return (
        SqrtGGNExact(subsampling=subsampling)
        if mc_samples == 0
        else SqrtGGNMC(subsampling=subsampling, mc_samples=mc_samples)
    )
