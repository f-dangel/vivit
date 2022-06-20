"""Contains a class that interfaces with BackPACK to compute damped Newton steps."""

import math
from typing import Callable, Dict, List, Optional, Tuple

from backpack.extensions import BatchGrad, SqrtGGNExact, SqrtGGNMC
from backpack.extensions.backprop_extension import BackpropExtension
from torch import Tensor, einsum
from torch.nn import Module

from vivit.linalg.utils import get_hook_store_batch_size
from vivit.optim.directional_derivatives import DirectionalDerivativesComputation
from vivit.optim.utils import get_sqrt_ggn_extension
from vivit.utils import delete_savefield
from vivit.utils.checks import (
    check_key_exists,
    check_subsampling_unique,
    check_unique_params,
)
from vivit.utils.gram import partial_contract, reshape_as_square
from vivit.utils.hooks import ParameterGroupsHook


class DirectionalDampedNewtonComputation:
    """Interfaces with BackPACK to compute directionally damped Newton steps."""

    def __init__(
        self,
        subsampling_grad: Optional[List[int]] = None,
        subsampling_ggn: Optional[List[int]] = None,
        mc_samples_ggn: Optional[int] = 0,
        verbose: Optional[bool] = False,
    ):
        check_subsampling_unique(subsampling_grad)
        check_subsampling_unique(subsampling_ggn)

        self._mc_samples_ggn = mc_samples_ggn

        if self._mc_samples_ggn != 0:
            assert mc_samples_ggn == 1
            self._extension_cls_ggn = SqrtGGNMC
        else:
            self._extension_cls_ggn = SqrtGGNExact

        self._extension_cls_grad = BatchGrad
        self._savefield_grad = self._extension_cls_grad().savefield
        self._subsampling_grad = subsampling_grad

        self._savefield_ggn = self._extension_cls_ggn().savefield
        self._subsampling_ggn = subsampling_ggn

        self._verbose = verbose

        # filled via side effects during update step computation, keys are group ids
        self._gammas = {}
        self._lambdas = {}
        self._batch_size = {}
        self._newton_steps: Dict[int, Tuple[Tensor]] = {}

    def get_result(self, group: Dict) -> Tuple[Tensor]:
        group_id = id(group)
        try:
            return self._newton_steps[group_id]
        except KeyError as e:
            raise KeyError("No results available for this group") from e

    def get_extensions(self) -> List[BackpropExtension]:
        return [
            self._extension_cls_grad(subsampling=self._subsampling_grad),
            get_sqrt_ggn_extension(
                subsampling=self._subsampling_ggn, mc_samples=self._mc_samples_ggn
            ),
        ]

    def get_extension_hook(self, param_groups: List[Dict]) -> Callable[[Module], None]:
        self._check_param_groups(param_groups)
        hook_store_batch_size = get_hook_store_batch_size(
            param_groups, self._batch_size, verbose=self._verbose
        )

        param_computation = lambda hook, param: self._param_computation(  # noqa: E731
            hook, param, self._savefield_ggn, self._savefield_grad, self._verbose
        )
        group_hook = lambda hook, accumulation, group: self._group_hook(  # noqa: E731
            hook,
            accumulation,
            group,
            self._batch_size,
            self._savefield_ggn,
            self._newton_steps,
            self._verbose,
        )
        accumulate = lambda hook, existing, update: self._accumulate(  # noqa: E731
            hook, existing, update, self._verbose
        )

        hook = ParameterGroupsHook.from_functions(
            param_groups, param_computation, group_hook, accumulate
        )

        def extension_hook(module: Module):
            if self._verbose:
                print(f"Extension hook on module {id(module)} {module}")
            hook_store_batch_size(module)
            hook(module)

        if self._verbose:
            print("ID map groups → params")
            for group in param_groups:
                print(f"{id(group)} → {[id(p) for p in group['params']]}")

        return extension_hook

    @staticmethod
    def _param_computation(
        hook: ParameterGroupsHook,
        param: Tensor,
        savefield_ggn: str,
        savefield_grad: str,
        verbose: bool,
    ) -> Dict[str, Tensor]:
        V = getattr(param, savefield_ggn)
        g = getattr(param, savefield_grad)

        if verbose:
            print(f"Param {id(param)}: Compute V_t_V and V_t_g_n")

        result = {
            "V_t_V": partial_contract(V, V, start_dims=(2, 2)),
            "V_t_g_n": partial_contract(V, g, start_dims=(2, 1)),
        }

        delete_savefield(param, savefield_grad, verbose=verbose)

        return result

    @staticmethod
    def _group_hook(
        hook: ParameterGroupsHook,
        accumulation: Dict[str, Tensor],
        group: Dict,
        batch_size: Dict[int, int],
        savefield_ggn: str,
        newton_steps: Dict[int, Tuple[Tensor]],
        verbose: bool,
    ):
        group_id = id(group)
        N = batch_size.pop(group_id)
        N_ggn = accumulation["V_t_V"].shape[1]

        # compensate scaling from BackPACK and subsampling
        V_correction = math.sqrt(N / N_ggn)
        gram_mat = V_correction**2 * accumulation.pop("V_t_V")
        C = gram_mat.shape[0]

        if verbose:
            print(f"Group {group_id}: Eigen-decompose Gram matrix")
        evals, evecs = reshape_as_square(gram_mat).symeig(eigenvectors=True)

        keep = group["criterion"](evals)
        if verbose:
            before, after = len(evals), len(keep)
            print(f"Group {group_id}: Filter directions ({before} → {after})")
        evals, evecs = evals[keep], evecs[:, keep]

        if verbose:
            print(f"Group {group_id}: Compute gammas")
        # compensate scaling from BackPACK and subsampling
        V_t_g_n = (
            V_correction
            * N
            * accumulation.pop("V_t_g_n").flatten(start_dim=0, end_dim=1)
        )
        gammas = einsum("in,id->nd", V_t_g_n, evecs) / evals.sqrt()

        if verbose:
            print(f"Group {group_id}: Compute lambdas")
        # compensate scaling from BackPACK and subsampling
        V_n_T_V_e_d = math.sqrt(N_ggn) * einsum(
            "cni,id->cnd", gram_mat.flatten(start_dim=2), evecs
        )
        lambdas = (V_n_T_V_e_d**2).sum(0) / evals

        # compute coefficients
        damping = group["damping"]
        coefficients = (
            -gammas.mean(0)
            / (lambdas.mean(0) + damping(evals, evecs, gammas, lambdas))
            / evals.sqrt()
        )

        # weight in Gram space, then transform to parameter space
        print(coefficients.shape)
        print(evecs.shape)
        v = einsum("id,d->i", evecs, coefficients)

        # Multiply by VT
        # compensate scaling
        v = v * V_correction

        params = group["params"]
        v = v.reshape(C, N_ggn)
        VTv = [
            einsum("cn,cn...->...", v, getattr(param, savefield_ggn))
            for param in params
        ]

        # clean up V
        for param in params:
            delete_savefield(param, savefield_ggn, verbose=verbose)

        newton_steps[group_id] = VTv

    @staticmethod
    def _accumulate(
        hook: ParameterGroupsHook,
        existing: Dict[str, Tensor],
        update: Dict[str, Tensor],
        verbose: bool,
    ):
        for key in existing.keys():
            if verbose:
                print(f"Accumulate dot product {key}")
            existing[key].add_(update[key])

        return existing

    @staticmethod
    def _check_param_groups(param_groups: List[Dict]):
        """Check if parameter groups satisfy the required format.

        Args:
            param_groups: Parameter groups that define the GGN block structure.
        """
        check_key_exists(param_groups, "params")
        check_key_exists(param_groups, "criterion")
        check_key_exists(param_groups, "damping")
        check_unique_params(param_groups)
