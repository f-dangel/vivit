"""Compute GGN eigenvalues during backpropagation."""

from typing import Any, Callable, Dict, List

from backpack.extensions.backprop_extension import BackpropExtension
from torch import Tensor
from torch.nn import Module, Parameter

from vivit.linalg.utils import get_hook_store_batch_size, get_vivit_extension
from vivit.utils.gram import reshape_as_square
from vivit.utils.hooks import ParameterGroupsHook


class EigvalshComputation:
    """Computes GGN eigenvalues during backpropagation via ``G = V Vᵀ``.

    This class provides two main functions for usage with BackPACK:

    - ``get_extension`` sets up the extension for a ``with backpack(...)`` context.
    - ``get_extension_hook`` sets up the hook for a ``with backpack(...)`` context.

    GGN eigenvalues will be stored as values of the dictionary ``self._evals`` with
    key corresponding to the parameter group id.
    """

    def __init__(
        self, subsampling: List[int] = None, mc_samples: int = 0, verbose=False
    ):
        """Store indices of samples used for each task.

        Assumes that the loss function uses ``reduction = 'mean'``.

        Args:
            subsampling: Sample indices used for the compution. Default ``None`` (all).
            mc_samples: Number of Monte-Carlo samples to approximate the loss Hessian.
                Default: ``0`` (exact loss Hessian).
            verbose: Turn on verbose mode. Default: ``False``.
        """
        self._subsampling = subsampling
        self._mc_samples = mc_samples
        self._verbose = verbose
        self._savefield = self.get_extension().savefield

        # filled via side effects during backpropagation, keys are group ids
        self._batch_size: Dict[int, int] = {}
        self._evals: Dict[int, Tensor] = {}

    def get_extension(self) -> BackpropExtension:
        """Instantiate the extension for a backward pass with BackPACK.

        Returns:
            Extension passed to a ``with backpack(...)`` context.
        """
        return get_vivit_extension(self._subsampling, self._mc_samples)

    def get_extension_hook(
        self,
        param_groups: List[Dict],
        keep_backpack_buffers: bool = False,
        keep_batch_size: bool = False,
    ) -> Callable[[Module], None]:
        """Return hook to be executed right after a BackPACK extension during backprop.

        This hook computes GGN eigenvalues during backpropagation and stores them under
        ``self._evals`` under the group id.

        Args:
            param_groups: Parameter group list as required by a
                ``torch.optim.Optimizer``. Specifies the block structure.
            keep_backpack_buffers: Keep buffers from used BackPACK extensions during
                backpropagation. Default: ``False``.
            keep_batch_size: Keep batch size stored under ``self._batch_size``.
                Default: ``False``.

        Returns:
            Hook function for a ``with backpack(...)`` context.
        """
        hook_store_batch_size = get_hook_store_batch_size(
            param_groups, self._batch_size, verbose=self._verbose
        )

        param_computation = self.get_param_computation(keep_backpack_buffers)
        group_hook = self.get_group_hook(keep_batch_size)
        accumulate = self.get_accumulate()

        hook = ParameterGroupsHook.from_functions(
            param_groups, param_computation, group_hook, accumulate
        )

        def extension_hook(module: Module):
            """Extension hook executed right after BackPACK extensions during backprop.

            Chains together all the required computations.

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

    def get_param_computation(
        self, keep_backpack_buffers: bool
    ) -> Callable[[ParameterGroupsHook, Parameter], Tensor]:
        """Set up the ``param_computation`` function of the ``ParameterGroupsHook``.

        Args:
            keep_backpack_buffers: Do not delete BackPACK buffers during
                backpropagation. If ``False``, they will be freed.

        Returns:
            Function that can be bound to a ``ParameterGroupsHook`` instance. Performs
            an action on each group parameter. The results are later accumulated.
        """
        verbose = self._verbose
        savefield = self._savefield

        def param_computation(self: ParameterGroupsHook, param: Parameter) -> Tensor:
            """Compute the Gram matrix and delete BackPACK buffers if specified.

            Args:
                self: Group hook to which this function will be bound.
                param: Parameter of a neural net.

            Returns:
                Non-square Gram matrix.
            """
            gram_mat = getattr(param, savefield)["gram_mat"]()

            if not keep_backpack_buffers:
                if verbose:
                    print(f"Param {id(param)}: Delete '{savefield}'")
                delattr(param, savefield)

            return gram_mat

        return param_computation

    def get_accumulate(self) -> Callable[[ParameterGroupsHook, Tensor, Tensor], Tensor]:
        """Set up the ``accumulate`` function of the ``ParameterGroupsHook``.

        Returns:
            Function that can be bound to a ``ParameterGroupsHook`` instance.
            Accumulates the parameter computations.
        """

        def accumulate(
            self: ParameterGroupsHook, existing: Tensor, update: Tensor
        ) -> Tensor:
            """Update existing Gram matrix with that of the parameter.

            Args:
                self: Group hook to which this function will be bound.
                existing: Previously accumulated Gram matrix.
                update: Gram matrix from current parameter.

            Returns:
                Updated Gram matrix.
            """
            return existing + update

        return accumulate

    def get_group_hook(
        self, keep_batch_size: bool
    ) -> Callable[[ParameterGroupsHook, Tensor, Dict[str, Any]], None]:
        """Set up the ``group_hook`` function of the ``ParameterGroupsHook``.

        Args:
            keep_batch_size: Keep batch size stored in ``self._batch_size``. Delete
                if ``True``.

        Returns:
            Function that can be bound to a ``ParameterGroupsHook`` instance. Performs
            an action on the accumulated results for a group.
        """
        batch_sizes = self._batch_size
        subsampling = self._subsampling
        evals = self._evals
        verbose = self._verbose

        def group_hook(self: ParameterGroupsHook, accumulation: Tensor, group: Dict):
            """Fix Gram matrix scale from sub-sampling and compute its eigenvalues.

            Args:
                self: Group hook to which this function will be bound.
                accumulation: Accumulated Gram matrix.
                group: Parameter group of a ``torch.optim.Optimizer``.
            """
            group_id = id(group)

            if keep_batch_size:
                batch_size = batch_sizes[group_id]
            else:
                if verbose:
                    print(f"Group {id(group)}: Delete 'batch_size'")
                batch_size = batch_sizes.pop(group_id)

            gram_mat = reshape_as_square(accumulation)

            # correct scale
            if subsampling is not None:
                gram_mat *= batch_size / len(subsampling)

            gram_evals, _ = gram_mat.symeig(eigenvectors=False)

            if verbose:
                print(f"Group {id(group)}: Store 'gram_evals'")
            evals[group_id] = gram_evals

        return group_hook
