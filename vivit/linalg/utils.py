"""Utility functions for ``vivit.linalg``."""

from typing import Callable, Dict, List, Union

from torch import Tensor, einsum
from torch.nn import Module

from vivit.extensions.secondorder.vivit import ViViTGGNExact, ViViTGGNMC


def get_vivit_extension(
    subsampling: Union[None, List[int]], mc_samples: int
) -> Union[ViViTGGNMC, ViViTGGNExact]:
    """Instantiate ``ViViT{Exact, MC} extension.

    Args:
        subsampling: Indices of active samples.
        mc_samples: Number of MC-samples to approximate the loss Hessian. ``0``
            uses the exact loss Hessian.

    Returns:
        Instantiated ViViT extension.
    """
    return (
        ViViTGGNExact(subsampling=subsampling)
        if mc_samples == 0
        else ViViTGGNMC(subsampling=subsampling, mc_samples=mc_samples)
    )


def get_hook_store_batch_size(
    param_groups: List[Dict], destination: Dict[int, int], verbose: bool = False
) -> Callable[[Module], None]:
    """Create extension hook that stores the batch size during backpropagation.

    Args:
        param_groups: Parameter group list from a ``torch.optim.Optimizer``.
        destination: Dictionary where the batch size will be saved to.
        verbose: Turn on verbose mode. Default: ``False``.

    Returns:
        Hook function to hand into a ``with backpack(...)`` context. Stores the
        batch size in the destination dictionary. under for each group.
    """

    def hook_store_batch_size(module: Module):
        """Store batch size internally. Modifies ``self._batch_size``.

        Args:
            module: The module on which the hook is executed.
        """
        if destination == {}:
            batch_axis = 0
            batch_size = module.input0.shape[batch_axis]

            for group in param_groups:
                group_id = id(group)

                if verbose:
                    print(f"Group {group_id}: Store 'batch_size'")

                destination[group_id] = batch_size

    return hook_store_batch_size


def normalize(tensors: List[Tensor]):
    """Normalize stacked vectors in parameter format (inplace).

    Args:
        tensors: Stacked vectors along the first dimension in parameter format.
    """
    inv_norm = 1 / sum(einsum("i...->i", t**2) for t in tensors).sqrt()

    for idx in range(len(tensors)):
        tensors[idx] = einsum("i,i...->i...", inv_norm, tensors[idx])
