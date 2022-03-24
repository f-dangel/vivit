"""Compute GGN eigenvalues during backpropagation."""

from typing import Any, Callable, Dict, List

from backpack.extensions.backprop_extension import BackpropExtension
from torch import Tensor
from torch.nn import Module, Parameter

from vivit.linalg.utils import get_hook_store_batch_size, get_vivit_extension
from vivit.utils import delete_savefield
from vivit.utils.checks import (
    check_key_exists,
    check_subsampling_unique,
    check_unique_params,
)
from vivit.utils.gram import reshape_as_square
from vivit.utils.hooks import ParameterGroupsHook


class EigvalshComputation:
    """Provide BackPACK extension and hook to compute GGN eigenvalues."""

    def __init__(
        self, subsampling: List[int] = None, mc_samples: int = 0, verbose: bool = False
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
        """
        check_subsampling_unique(subsampling)

        self._subsampling = subsampling
        self._mc_samples = mc_samples
        self._verbose = verbose
        self._savefield = self.get_extension().savefield

        # filled via side effects during backpropagation, keys are group ids
        self._batch_size: Dict[int, int] = {}
        self._evals: Dict[int, Tensor] = {}

    def get_result(self, group: Dict) -> Tensor:
        """Return eigenvalues of a GGN block after the backward pass.

        Args:
            group: Parameter group that defines the GGN block.

        Returns:
            One-dimensional tensor containing the block's eigenvalues (ascending order).

        Raises:
            KeyError: If there are no results for the group.
        """
        try:
            return self._evals[id(group)]
        except KeyError as e:
            raise KeyError("No results available for this group") from e

    def get_extension(self) -> BackpropExtension:
        """Instantiate the BackPACK extension for computing GGN eigenvalues.

        Returns:
            BackPACK extension to compute eigenvalues that should be passed to the
            :py:class:`with backpack(...) <backpack.backpack>` context.
        """
        return get_vivit_extension(self._subsampling, self._mc_samples)

    def get_extension_hook(self, param_groups: List[Dict]) -> Callable[[Module], None]:
        """Instantiates the BackPACK extension hook to compute GGN eigenvalues.

        Args:
            param_groups: Parameter groups list as required by a
                ``torch.optim.Optimizer``. Specifies the block structure: Each group
                must specify the ``'params'`` key which contains a list of the
                parameters that form a GGN block. Examples:

                - ``[{'params': list(p for p in model.parameters()}]`` computes
                  eigenvalues for the full GGN (one block).
                - ``[{'params': [p]} for p in model.parameters()]`` computes
                  eigenvalues for each block of a per-parameter block-diagonal GGN
                  approximation.

        Returns:
            BackPACK extension hook to compute eigenvalues that should be passed to the
            :py:class:`with backpack(...) <backpack.backpack>` context. The hook
            computes GGN eigenvalues during backpropagation and stores them internally
            (under ``self._evals``).
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
        self,
    ) -> Callable[[ParameterGroupsHook, Parameter], Tensor]:
        """Set up the ``param_computation`` function of the ``ParameterGroupsHook``.

        Returns:
            Function that can be bound to a ``ParameterGroupsHook`` instance. It
            computes the Gram matrix for a parameter and deletes the BackPACK buffer.
        """
        verbose = self._verbose
        savefield = self._savefield

        def param_computation(self: ParameterGroupsHook, param: Parameter) -> Tensor:
            """Compute the Gram matrix and delete BackPACK buffers.

            Args:
                self: Group hook to which this function will be bound.
                param: Parameter of a neural net.

            Returns:
                Non-square Gram matrix.
            """
            gram_mat = getattr(param, savefield)["gram_mat"]()
            delete_savefield(param, savefield, verbose=verbose)

            return gram_mat

        return param_computation

    def get_accumulate(self) -> Callable[[ParameterGroupsHook, Tensor, Tensor], Tensor]:
        """Set up the ``accumulate`` function of the ``ParameterGroupsHook``.

        Returns:
            Function that can be bound to a ``ParameterGroupsHook`` instance.
            Accumulates the parameter Gram matrices.
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
        self,
    ) -> Callable[[ParameterGroupsHook, Tensor, Dict[str, Any]], None]:
        """Set up the ``group_hook`` function of the ``ParameterGroupsHook``.

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

    @staticmethod
    def _check_param_groups(param_groups: List[Dict]):
        """Check if parameter groups satisfy the required format.

        Args:
            param_groups: Parameter groups that define the GGN block structure.
        """
        check_key_exists(param_groups, "params")
        check_unique_params(param_groups)
