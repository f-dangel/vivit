"""Contains extension for the linear layer used by ``ViViTGGN{Exact, MC}``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, Tuple, Union

from backpack.core.derivatives.linear import LinearDerivatives
from backpack.utils.subsampling import subsample
from torch import Tensor, einsum
from torch.nn import Module

from vivit.extensions.secondorder.vivit.base import ViViTGGNBaseModule
from vivit.utils.gram import pairwise_dot

if TYPE_CHECKING:
    from vivit.extensions.secondorder.vivit import ViViTGGNExact, ViViTGGNMC


class ViViTGGNLinear(ViViTGGNBaseModule):
    """``ViViTGGN{Exact, MC}`` extension for ``torch.nn.Linear`` module."""

    def __init__(self):
        """Pass derivatives for ``torch.nn.Linear`` module."""
        super().__init__(LinearDerivatives(), params=["bias", "weight"])

        # TODO Implement optimized Gram matrix for additional input dimensions
        self.weight_additional = super()._make_param_function("weight")

    def weight(
        self,
        ext: Union[ViViTGGNExact, ViViTGGNMC],
        module: Module,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        backproped: Tensor,
    ) -> Dict[str, Callable]:  # noqa: D102

        if self.derivatives._get_additional_dims(module):
            return self.weight_additional(ext, module, g_inp, g_out, backproped)

        s = backproped
        z = subsample(module.input0, dim=0, subsampling=ext.get_subsampling())

        def V_mat_prod(mat: Tensor) -> Tensor:
            """Multiply ``V`` to multiple vectors. Use structure of the Linear layer.

            Args:
                mat: Stacked vectors to which ``V`` will be applied

            Returns:
                Result of applying ``V``. Shape ``[mat.shape[0], *param.shape]``.
            """
            return einsum("cno,vcn,ni->voi", s, mat, z)

        def V_t_mat_prod(mat: Tensor) -> Tensor:
            """Multiply ``Vᵀ`` to multiple vectors. Use structure in the linear layer.

            Args:
                mat: Stacked vectors to which ``Vᵀ`` will be applied

            Returns:
                Result of applying ``Vᵀ``. Shape ``[mat.shape[0], C, N]``.
            """
            return einsum("cno,voi,ni->vcn", s, mat, z)

        def gram_mat() -> Tensor:
            """Evaluate the Gram matrix. Use structure of the Linear layer.

            Returns:
                Gram matrix. Shape ``[C, N, C, N]``.
            """
            s_second_moment = pairwise_dot(s, start_dim=2, flatten=False)
            z_second_moment = pairwise_dot(z, start_dim=1, flatten=False)

            return einsum("nm,cndm->cndm", z_second_moment, s_second_moment)

        return {
            "V_mat_prod": V_mat_prod,
            "V_t_mat_prod": V_t_mat_prod,
            "gram_mat": gram_mat,
        }
