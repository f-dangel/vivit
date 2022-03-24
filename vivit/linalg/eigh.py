"""Compute GGN eigenvalues and eigenvectors during backpropagation."""

from typing import Any, Callable, Dict, List, Tuple
from warnings import warn

from backpack.extensions.backprop_extension import BackpropExtension
from torch import Tensor
from torch.nn import Module, Parameter

from vivit.linalg.utils import get_hook_store_batch_size, get_vivit_extension, normalize
from vivit.utils import delete_savefield
from vivit.utils.checks import (
    check_key_exists,
    check_subsampling_unique,
    check_unique_params,
)
from vivit.utils.gram import reshape_as_square
from vivit.utils.hooks import ParameterGroupsHook


class EighComputation:
    """Provide BackPACK extension and hook to compute GGN eigenpairs."""

    def __init__(
        self,
        subsampling: List[int] = None,
        mc_samples: int = 0,
        verbose: bool = False,
        warn_small_eigvals: float = 1e-4,
    ):
        """Specify GGN approximations. Use no approximations by default.

        Note:
            The loss function must use ``reduction = 'mean'``.

        Args:
            subsampling: Indices of samples used for GGN curvature sub-sampling.
                ``None`` (equivalent to ``list(range(batch_size))``) uses all mini-batch
                samples. Defaults to ``None`` (no curvature sub-sampling).
            mc_samples: If ``0``, don't Monte-Carlo (MC) approximate the GGN. Otherwise,
                specifies the number of MC samples used to approximate the
                backpropagated loss Hessian. Default: ``0`` (no MC approximation).
            verbose: Turn on verbose mode. If enabled, this will print what's happening
                during backpropagation to command line (consider it a debugging tool).
                Defaults to ``False``.
            warn_small_eigvals: The eigenvector computation breaks down for numerically
                small eigenvalues close to zero. This variable triggers a user warning
                when attempting to compute eigenvectors for eigenvalues whose absolute
                value is smaller. Defaults to ``1e-4``. You can disable the warning by
                setting it to ``0`` (not recommended).
        """
        check_subsampling_unique(subsampling)

        self._subsampling = subsampling
        self._mc_samples = mc_samples
        self._verbose = verbose
        self._savefield = self.get_extension().savefield
        self._warn_small_eigvals = warn_small_eigvals

        # filled via side effects during update step computation, keys are group ids
        self._batch_size: Dict[int, int] = {}
        self._evals: Dict[int, Tensor] = {}
        self._evecs: Dict[int, List[Tensor]] = {}

    def get_result(self, group: Dict) -> Tuple[Tensor, List[Tensor]]:
        """Return eigenvalues and eigenvectors of a GGN block after the backward pass.

        Args:
            group: Parameter group that defines the GGN block.

        Returns:
            One-dimensional tensor containing the block's eigenvalues and eigenvectors
            in parameter list format with leading axis for the eigenvectors.

            Example: Let ``evals, evecs`` denote the returned variables. Let
            ``group['params'] = [p1, p2]`` consist of two parameters. Then, ``evecs``
            contains tensors of shape
            ``[(evals.numel(), *p1.shape), (evals.numel(), *p2.shape)]``. For an
            eigenvalue ``evals[k]``, the corresponding eigenvector (in list format) is
            ``[vecs[k] for vecs in evecs]`` and its tensors are of shape
            ``[p1.shape, p2.shape]``.

        Raises:
            KeyError: If there are no results for the group.
        """
        group_id = id(group)
        try:
            return self._evals[group_id], self._evecs[group_id]
        except KeyError as e:
            raise KeyError("No results available for this group") from e

    def get_extension(self) -> BackpropExtension:
        """Instantiate the BackPACK extension for computing GGN eigenpairs.

        Returns:
            BackPACK extension to compute eigenvalues that should be passed to the
            :py:class:`with backpack(...) <backpack.backpack>` context.
        """
        return get_vivit_extension(self._subsampling, self._mc_samples)

    def get_extension_hook(self, param_groups: List[Dict]) -> Callable[[Module], None]:
        """Instantiates the BackPACK extension hook to compute GGN eigenpairs.

        Args:
            param_groups: Parameter groups list as required by a
                ``torch.optim.Optimizer``. Specifies the block structure: Each group
                must specify the ``'params'`` key which contains a list of the
                parameters that form a GGN block, and a ``'criterion'`` entry that
                specifies a filter function to select eigenvalues for the eigenvector
                computation (details below).

                Examples for ``'params'``:

                - ``[{'params': list(p for p in model.parameters()}]`` uses the full
                  GGN (one block).
                - ``[{'params': [p]} for p in model.parameters()]`` uses a per-parameter
                  block-diagonal GGN approximation.

                The function specified under ``'criterion'`` is a
                ``Callable[[Tensor], List[int]]``. It receives the eigenvalues (in
                ascending order) and returns the indices of eigenvalues whose
                eigenvectors should be computed. Examples:

                - ``{'criterion': lambda evals: [evals.numel() - 1]}`` discards all
                  eigenvalues except for the largest.
                - ``{'criterion': lambda evals: list(range(evals.numel()))}`` computes
                  eigenvectors for all Gram matrix eigenvalues.

        Returns:
            BackPACK extension hook to compute eigenpairs that should be passed to the
            :py:class:`with backpack(...) <backpack.backpack>` context. The hook
            computes GGN eigenpairs during backpropagation and stores them internally
            (under ``self._evals`` and ``self._evecs``).
        """
        self._check_param_groups(param_groups)
        hook_store_batch_size = get_hook_store_batch_size(
            param_groups, self._batch_size, verbose=self._verbose
        )

        param_computation = self.get_param_computation()
        group_hook = self.get_group_hook()
        accumulate = self.get_accumulate()

        hook = ParameterGroupsHook.from_functions(
            param_groups, param_computation, group_hook, accumulate
        )

        def extension_hook(module: Module):
            """Compute GGN eigenpairs when passed as BackPACK extension hook.

            Args:
                module: Layer on which the hook is executed.
            """
            if self._verbose:
                print(f"Extension hook on module {id(module)} {module}")
            hook_store_batch_size(module)
            hook(module)

        if self._verbose:
            print("ID map groups → params")
            for group in param_groups:
                print(f"{id(group)} → {[id(p) for p in group['params']]}")

        return extension_hook

    def get_param_computation(self) -> Callable[[ParameterGroupsHook, Parameter], None]:
        """Set up the ``param_computation`` function of the ``ParameterGroupsHook``.

        Returns:
            Function that can be bound to a ``ParameterGroupsHook`` instance. Performs
            an action on each group parameter. The results are later accumulated.
        """

        def param_computation(self: ParameterGroupsHook, param: Parameter):
            """No computation required for a single parameter.

            Args:
                self: Group hook to which this function will be bound.
                param: Parameter of a neural net.
            """
            pass

        return param_computation

    def get_accumulate(self) -> Callable[[ParameterGroupsHook, None, None], None]:
        """Set up the ``accumulate`` function of the ``ParameterGroupsHook``.

        Returns:
            Function that can be bound to a ``ParameterGroupsHook`` instance.
            Accumulates the parameter computations.
        """

        def accumulate(self: ParameterGroupsHook, existing: None, update: None) -> None:
            """No accumulation required for a group.

            Args:
                self: Group hook to which this function will be bound.
                existing: Dictionary containing the accumulated results so far.
                update: Dictionary containing results for a parameter.
            """
            pass

        return accumulate

    def get_group_hook(
        self,
    ) -> Callable[[ParameterGroupsHook, None, Dict[str, Any]], None]:
        """Set up the ``group_hook`` function of the ``ParameterGroupsHook``.

        Returns:
            Function that can be bound to a ``ParameterGroupsHook`` instance. Performs
            an action on the accumulated results over parameters for a group.
        """
        batch_sizes = self._batch_size
        subsampling = self._subsampling
        savefield = self._savefield
        evals = self._evals
        evecs = self._evecs
        verbose = self._verbose
        warn_small_eigvals = self._warn_small_eigvals

        def group_hook(
            self: ParameterGroupsHook, accumulation: None, group: Dict[str, Any]
        ) -> None:
            """Compute & accumulate Gram mat, decompose & transform to parameter space.

            Args:
                self: Group hook to which this function will be bound.
                accumulation: Accumulated results from parameter computations.
                group: Parameter group of a ``torch.optim.Optimizer``.
            """
            group_id = id(group)

            if verbose:
                print(f"Group {id(group)}: Delete 'batch_size'")
            batch_size = batch_sizes.pop(group_id)

            # Compute & accumulate Gram mat
            gram_mat = 0.0

            for param in group["params"]:
                gram_mat += getattr(param, savefield)["gram_mat"]()

            # correct scale
            if subsampling is not None:
                gram_mat *= batch_size / len(subsampling)

            gram_evals, gram_evecs = reshape_as_square(gram_mat).symeig(
                eigenvectors=True
            )

            keep = group["criterion"](gram_evals)
            gram_evals, gram_evecs = gram_evals[keep], gram_evecs[:, keep]

            # warn about numerical instabilities
            if (gram_evals.abs() < warn_small_eigvals).any():
                warn(
                    "Some eigenvectors have small eigenvalues."
                    + " Their parameter space transformation is numerically unstable."
                    + " This can spoil orthogonality of eigenvectors."
                    + " Maybe use a more restrictive eigenvalue filter criterion."
                )

            # make eigenvectors selectable via first axis
            gram_evecs = gram_evecs.transpose(0, 1).reshape(-1, *gram_mat.shape[:2])

            group_evecs = []
            for param in group["params"]:
                group_evecs.append(getattr(param, savefield)["V_mat_prod"](gram_evecs))
                delete_savefield(param, savefield, verbose=verbose)

            normalize(group_evecs)

            evals[group_id] = gram_evals
            evecs[group_id] = group_evecs

        return group_hook

    @staticmethod
    def _check_param_groups(param_groups: List[Dict]):
        """Check if parameter groups satisfy the required format.

        Each group must specify ``'params'`` and ``'criterion'``. Parameters can
        only belong to one group.

        Args:
            param_groups: Parameter groups that define the GGN block structure and
                the selection criterion for which eigenvalues to compute eigenvectors.
        """
        check_key_exists(param_groups, "params")
        check_key_exists(param_groups, "criterion")
        check_unique_params(param_groups)
