"""Contains a class that interfaces with BackPACK to compute damped Newton steps."""

import math
from typing import Callable, Dict, List, Optional, Tuple
from warnings import warn

from backpack.extensions import BatchGrad, SqrtGGNExact, SqrtGGNMC
from backpack.extensions.backprop_extension import BackpropExtension
from torch import Tensor, einsum
from torch.nn import Module

from vivit.linalg.utils import get_hook_store_batch_size
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
    r"""Provides BackPACK extension and hook for directionally damped Newton steps.

    Let :math:`\{e_k\}_{k = 1, \dots, K}` denote the :math:`K` eigenvectors
    used for the Newton step. Let :math:`\gamma_k, \lambda_k` denote the first-
    and second-order derivatives along these eigenvectors. The direcionally
    damped Newton step :math:`s` is

    .. math::
        s = \sum_{k=1}^K \frac{- \gamma_k}{\lambda_k + \delta_k} e_k

    Here, :math:`\delta_k` is the damping along direction :math:`k`. The exact
    Newton step has zero damping.
    """

    def __init__(
        self,
        subsampling_grad: Optional[List[int]] = None,
        subsampling_ggn: Optional[List[int]] = None,
        mc_samples_ggn: Optional[int] = 0,
        verbose: Optional[bool] = False,
        warn_small_eigvals: float = 1e-4,
    ):
        """Specify GGN and gradient approximations. Use no approximations by default.

        Note:
            The loss function must use ``reduction = 'mean'``.

        Args:
            subsampling_grad: Indices of samples used for gradient sub-sampling.
                ``None`` (equivalent to ``list(range(batch_size))``) uses all mini-batch
                samples to compute directional gradients . Defaults to ``None`` (no
                gradient sub-sampling).
            subsampling_ggn: Indices of samples used for GGN curvature sub-sampling.
                ``None`` (equivalent to ``list(range(batch_size))``) uses all mini-batch
                samples to compute directions and directional curvatures. Defaults to
                ``None`` (no curvature sub-sampling).
            mc_samples_ggn: If ``0``, don't Monte-Carlo (MC) approximate the GGN
                (using the same samples to compute the directions and directional
                curvatures). Otherwise, specifies the number of MC samples used to
                approximate the backpropagated loss Hessian. Default: ``0`` (no MC
                approximation).
            verbose: Turn on verbose mode. If enabled, this will print what's happening
                during backpropagation to command line (consider it a debugging tool).
                Defaults to ``False``.
            warn_small_eigvals: The ``gamma`` computation and tranformation from Gram
                into parameter space break down for numerically small eigenvalues close
                to zero (need to divide by their square root). This variable triggers a
                user warning when attempting to compute damped Newton steps for
                eigenvalues whose absolute value is smaller. Defaults to ``1e-4``.
                You can disable the warning by setting it to ``0`` (not recommended).
        """
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
        self._warn_small_eigvals = warn_small_eigvals

        # filled via side effects during update step computation, keys are group ids
        self._batch_size = {}
        self._newton_steps: Dict[int, Tuple[Tensor]] = {}

    def get_result(self, group: Dict) -> Tuple[Tensor]:
        """Return a damped Newton step in parameter format.

        Must be called after the backward pass.

        Args:
            group: Parameter group that defines the GGN block.

        Returns:
            Damped Newton step in parameter format, i.e. the same format as
            ``group['params']``.

        Raises:
            KeyError: If there are no results for the group.
        """
        group_id = id(group)
        try:
            return self._newton_steps[group_id]
        except KeyError as e:
            raise KeyError("No results available for this group") from e

    def get_extensions(self) -> List[BackpropExtension]:
        """Instantiate the BackPACK extensions to compute a damped Newton step.

        Returns:
            BackPACK extensions, to compute a damped Newton step, that should be
            extracted and passed to the
            :py:class:`with backpack(...) <backpack.backpack>` context.
        """
        return [
            self._extension_cls_grad(subsampling=self._subsampling_grad),
            get_sqrt_ggn_extension(
                subsampling=self._subsampling_ggn, mc_samples=self._mc_samples_ggn
            ),
        ]

    def get_extension_hook(self, param_groups: List[Dict]) -> Callable[[Module], None]:
        r"""Instantiate BackPACK extension hook to compute a damped Newton step.

        Args:
            param_groups: Parameter groups list as required by a
                ``torch.optim.Optimizer``. Specifies the block structure: Each group
                must specify the ``'params'`` key which contains a list of the
                parameters that form a GGN block, a ``'criterion'`` entry that
                specifies a filter function to select eigenvalues as directions along
                which to compute the damped Newton step, and a ``'damping'`` entry that
                that contains a function that produces the directional dampings
                (details below).

                Examples for ``'params'``:

                - ``[{'params': list(p for p in model.parameters()}]`` uses the full
                  GGN (one block).
                - ``[{'params': [p]} for p in model.parameters()]`` uses a per-parameter
                  block-diagonal GGN approximation.

                The function specified under ``'criterion'`` is a
                ``Callable[[Tensor], List[int]]``. It receives the eigenvalues (in
                ascending order) and returns the indices of eigenvalues whose
                eigenvectors should be used as directions for the damped Newon step.
                Examples:

                - ``{'criterion': lambda evals: [evals.numel() - 1]}`` discards all
                  directions except for the leading eigenvector.
                - ``{'criterion': lambda evals: list(range(evals.numel()))}`` computes
                  damped Newton step along all Gram matrix eigenvectors.

                The function specified under ``'damping'`` is a
                ``Callable[[Tensor, Tensor, Tensor, Tensor]]``. It receives the
                eigenvalues, Gram eigenvectors, directional gradients and directional
                curvatures, and returns a tensor that contains the damping values
                for all directions. Example:

                - ``{'damping': lambda evals, evecs, gammas, lambdas:
                  ones_like(evals)}`` corresponds to constant damping
                  :math:`\delta_k = 1\ \forall k`.

        Returns:
            BackPACK extension hook, to compute damped Newton steps, that should be
            passed to the :py:class:`with backpack(...) <backpack.backpack>` context.
            The hook computes a damped Newton step during backpropagation and
            stores it internally (under ``self._newton_step``).

        """
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
            self._warn_small_eigvals,
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
        """Compute dot products for the parameter.

        A partially-evaluated form of this function can be bound to a
        ``ParameterGroupsHook.param_computation``.

        Args:
            hook: Group hook to which this function can be bound.
            param: Parameter of a neural net.
            savefield_ggn: Name under which the GGN square root is stored in param
            savefield_grad: Name under which individual gradients are stored in param
            verbose: Whether to print steps of the computation to command line.

        Returns:
            Dictionary containing the dot products ``"V_t_g_n"`` & ``"V_t_V"``.
        """
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
        newton_steps: Dict[int, List[Tensor]],
        verbose: bool,
        warn_small_eigvals: bool,
    ):
        """Compute damped Newton step for a group and store it in ``newton_steps``.

        A partially-evaluated form of this function can be bound to a
        ``ParameterGroupsHook.group_hook``.

        In detail, the steps are as follows:

        1. Compute the Gram matrix from the ``V_t_V`` dot product, compute its
           eigendecomposition. Filter the spectrum according to the parameter
           group's ``'criterion'`` entry.
        2. Compute the directional gradients (from ``V_t_g_n``) and curvatures (from
           the Gram matrix and its filtered eigenvectors).
        3. Compute the directional dampings using the parameter group's ``'damping'``
           entry.
        4. Compute the weighting coefficients for the directions. Weight them in Gram
           space, then transform the Newton step into parameter space by application
           of ``V``.

        Args:
            hook: Group hook to which this function will be bound.
            accumulation: Accumulated dot products.
            group: Parameter group of a ``torch.optim.Optimizer``.
            batch_size: Mapping from group id to batch size.
            savefield_ggn: Attribute under which the GGN square root is stored in a
                parameter.
            newton_steps: Dictionary in which the computed Newton step will be stored
                under the group's id.
            verbose: Whether to print steps of the computation to command line.
            warn_small_eigvals: Warn the user that eigenvalues are smalller than
                specified value.
        """
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

        # warn about numerical instabilities
        if (evals.abs() < warn_small_eigvals).any():
            warn(
                "Some eigenvalues are small. This can lead to numerical instabilities"
                + " in the directional gradients and the transformation into parameter"
                + " space because they require division by the eigenvalue square root."
                + " Maybe use a more restrictive eigenvalue filter criterion."
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
        v = einsum("id,d->i", evecs, coefficients)

        # Multiply by VT
        # compensate scaling
        v = v * V_correction

        params = group["params"]
        v = v.reshape(C, N_ggn)
        Vv: List[Tensor] = [
            einsum("cn,cn...->...", v, getattr(param, savefield_ggn))
            for param in params
        ]

        # clean up V
        for param in params:
            delete_savefield(param, savefield_ggn, verbose=verbose)

        newton_steps[group_id] = Vv

    @staticmethod
    def _accumulate(
        hook: ParameterGroupsHook,
        existing: Dict[str, Tensor],
        update: Dict[str, Tensor],
        verbose: bool,
    ) -> Dict[str, Tensor]:
        """Accumulate per-parameter directional derivative dot products.

        A partially-evaluated form of this function can be bound to a
        ``ParameterGroupsHook.group_hook``.

        Args:
            hook: Group hook to which this function will be bound.
            existing: Dictionary containing the so far accumulated scalar products.
            update: Dictionary containing the scalar product updates.
            verbose: Whether to print steps of the computation to command line.

        Returns:
            Updated scalar products.
        """
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
