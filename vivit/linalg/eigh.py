"""Compute GGN eigenvalues and eigenvectors during backpropagation."""

from typing import Any, Callable, Dict, List
from warnings import warn

from backpack.extensions.backprop_extension import BackpropExtension
from torch import Tensor
from torch.nn import Module, Parameter

from vivit.linalg.utils import get_hook_store_batch_size, get_vivit_extension, normalize
from vivit.utils.gram import reshape_as_square
from vivit.utils.hooks import ParameterGroupsHook


class EighComputation:
    """Computes GGN eigenvalues/eigenvectors during backpropagation via ``G = V Vᵀ``.

    This class provides two main functions for usage with BackPACK:

    - ``get_extension`` sets up the extension for a ``with backpack(...)`` context.
    - ``get_extension_hook`` sets up the hook for a ``with backpack(...)`` context.

    GGN eigenvalues/eigenvectors will be stored as values of the dictionaries
    ``self._evals``/``self._evecs`` with key corresponding to the parameter group id.
    """

    def __init__(
        self,
        subsampling: List[int] = None,
        mc_samples: int = 0,
        verbose=False,
        warn_small_eigvals: float = 1e-4,
    ):
        """Store indices of samples used for each task.

        Assumes that the loss function uses ``reduction = 'mean'``.

        Args:
            subsampling: Indices of samples used for the computation. Default ``None``
                uses the entire mini-batch.
            mc_samples: Number of Monte-Carlo samples to approximate the loss Hessian.
                Default of ``0`` uses the exact loss Hessian.
            verbose: Turn on verbose mode. Default: ``False``.
            warn_small_eigvals: Warns the user about numerical instabilities for small
                eigenvalues that have smaller magnitude. Default: ``1e-4``.
        """
        self._subsampling = subsampling
        self._mc_samples = mc_samples
        self._verbose = verbose
        self._savefield = self.get_extension().savefield
        self._warn_small_eigvals = warn_small_eigvals

        # filled via side effects during update step computation, keys are group ids
        self._batch_size: Dict[int, int] = {}
        self._evals: Dict[int, Tensor] = {}
        self._evecs: Dict[int, List[Tensor]] = {}

    def get_extension(self) -> BackpropExtension:
        """Instantiate the extension for a backward pass with BackPACK.

        Returns:
            Extension passed to a ``with backpack(..._)`` context.
        """
        return get_vivit_extension(self._subsampling, self._mc_samples)

    def get_extension_hook(
        self,
        param_groups: List[Dict],
        keep_backpack_buffers: bool = False,
        keep_batch_size: bool = False,
    ) -> Callable[[Module], None]:
        """Return hook to be executed right after a BackPACK extension during backprop.

        This hook computes GGN eigenvalues and eigenvectors during backpropagation and
        stores them under ``self._evals`` and ``self.evecs`` under the group id,
        respectively.

        Args:
            param_groups: Parameter group list like for a ``torch.optim.Optimizer``.
                Each group must have a 'criterion' entry which specifies a
                ``Callable[[Tensor], List[int]]`` that processes the eigenvalues and
                returns the indices of those that should be kept for the eigenvector
                computation.
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

        param_computation = self.get_param_computation()
        group_hook = self.get_group_hook(keep_backpack_buffers, keep_batch_size)
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
        self, keep_backpack_buffers: bool, keep_batch_size: bool
    ) -> Callable[[ParameterGroupsHook, None, Dict[str, Any]], None]:
        """Set up the ``group_hook`` function of the ``ParameterGroupsHook``.

        Args:
            keep_backpack_buffers: Keep BackPACK's buffers during backpropagation.
                Delete if ``True``.
            keep_batch_size: Keep batch size stored under ``self._batch_size``.
                Delete if ``True``.

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
            if keep_batch_size:
                batch_size = batch_sizes[group_id]
            else:
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

                if not keep_backpack_buffers:
                    if verbose:
                        print(f"Param {id(param)}: Delete '{savefield}'")
                    delattr(param, savefield)

            normalize(group_evecs)

            evals[group_id] = gram_evals
            evecs[group_id] = group_evecs

        return group_hook
