"""Contains base class for ``ViViTGGN{Exact, MC}`` module extensions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, List, Tuple, Union

from backpack.core.derivatives.basederivatives import BaseDerivatives
from backpack.extensions.mat_to_mat_jac_base import MatToJacMat
from torch import Tensor
from torch.nn import Module

from vivit.utils.ggn import Vmp
from vivit.utils.gram import mVp, pairwise_dot

if TYPE_CHECKING:
    from vivit.extensions.secondorder.vivit import ViViTGGNExact, ViViTGGNMC


class ViViTGGNBaseModule(MatToJacMat):
    """Base module extension for ``ViViTGGN{Exact, MC}``."""

    def __init__(self, derivatives: BaseDerivatives, params: List[str] = None):
        """Store parameter names and derivatives.

        Sets up methods that extract functions to
        - multiply a collection of vectors by ``V``
        - multiply a collection of vectors by ``Vᵀ``
        - evaluate the Gram matrix ``Vᵀ V``
        for the passed parameters, unless these methods are overwritten by a child
        class.

        Args:
            derivatives: derivatives object.
            params: List of parameter names. Defaults to None.
        """
        if params is not None:
            for param_str in params:
                if not hasattr(self, param_str):
                    setattr(self, param_str, self._make_param_function(param_str))

        super().__init__(derivatives, params=params)

    def _make_param_function(
        self, param_str: str
    ) -> Callable[
        [
            Union[ViViTGGNExact, ViViTGGNMC],
            Module,
            Tuple[Tensor],
            Tuple[Tensor],
            Tensor,
        ],
        Dict[str, Callable],
    ]:
        """Create a function that computes matrix-multiply and Gram matrix functions.

        Args:
            param_str: name of parameter

        Returns:
            Function that computes a dictionary containing the functions to perform
            matrix-multiplies and evaluate the Gram matrix.
        """

        def param_function(
            ext: Union[ViViTGGNExact, ViViTGGNMC],
            module: Module,
            g_inp: Tuple[Tensor],
            g_out: Tuple[Tensor],
            backproped: Tensor,
        ) -> Dict[str, Callable]:
            """Set up functions to multiply with ``V``, ``Vᵀ``, and evaluate ``Vᵀ V``.

            Args:
                ext: extension that is used
                module: module that performed forward pass
                g_inp: input gradient tensors
                g_out: output gradient tensors
                backproped: Backpropagated quantities from second-order extension.

            Returns:
                Dictionary containing the described functions.
            """
            V_t = self.derivatives.param_mjp(
                param_str,
                module,
                g_inp,
                g_out,
                backproped,
                sum_batch=False,
                subsampling=ext.get_subsampling(),
            )

            start_dim = 2

            def V_mat_prod(mat: Tensor) -> Tensor:
                """Multiply ``V`` to multiple vectors.

                Args:
                    mat: Stacked vectors to which ``V`` will be applied

                Returns:
                    Result of applying ``V``. Shape ``[mat.shape[0], *param.shape]``.
                """
                return Vmp(V_t, mat, start_dim)

            def V_t_mat_prod(mat: Tensor) -> Tensor:
                """Multiply ``Vᵀ`` to multiple vectors.

                Args:
                    mat: Stacked vectors to which ``Vᵀ`` will be applied

                Returns:
                    Result of applying ``Vᵀ``. Shape ``[C, N]``.
                """
                return mVp(V_t, mat, start_dim)

            def gram_mat() -> Tensor:
                """Evaluate the Gram matrix.

                Returns:
                    Gram matrix. Shape ``[C, N, C, N]``.
                """
                return pairwise_dot(V_t, start_dim=start_dim, flatten=False)

            return {
                "V_mat_prod": V_mat_prod,
                "V_t_mat_prod": V_t_mat_prod,
                "gram_mat": gram_mat,
            }

        return param_function
