"""Manage computations in Gram space through extension hooks."""

import math
import warnings
from typing import Callable, Dict, List, Optional

import torch
from backpack.extensions import BatchGrad, SqrtGGNExact, SqrtGGNMC
from backpack.extensions.backprop_extension import BackpropExtension
from torch.nn import Module

from vivit.utils.eig import stable_symeig
from vivit.utils.gram import partial_contract, reshape_as_square
from vivit.utils.hooks import ParameterGroupsHook
from vivit.utils.subsampling import is_subset, merge_extensions, sample_output_mapping


class DirectionalDerivativesComputation:
    """Provide BackPACK extension and hook for 1ˢᵗ/2ⁿᵈ-order directional derivatives.

    The directions are given by the GGN eigenvectors. First-order directional
    derivatives are denoted ``γ``, second-order directional derivatives as ``λ``.
    """

    def __init__(
        self,
        subsampling_grad: Optional[List[int]] = None,
        subsampling_ggn: Optional[List[int]] = None,
        mc_samples_ggn: Optional[int] = 0,
        verbose: Optional[bool] = False,
    ):
        """Specify GGN and gradient approximations. Use no approximations by default.

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
        """
        subsampling_directions = subsampling_ggn
        subsampling_first = subsampling_grad
        subsampling_second = subsampling_ggn

        if mc_samples_ggn != 0:
            assert mc_samples_ggn == 1
            extension_cls_directions = SqrtGGNMC
            extension_cls_second = SqrtGGNMC
        else:
            extension_cls_directions = SqrtGGNExact
            extension_cls_second = SqrtGGNExact

        self._extension_cls_first = BatchGrad
        self._savefield_first = self._extension_cls_first().savefield
        self._subsampling_first = subsampling_first

        self._extension_cls_second = extension_cls_second
        self._savefield_second = extension_cls_second().savefield
        self._subsampling_second = subsampling_second

        self._extension_cls_directions = extension_cls_directions
        self._savefield_directions = self._extension_cls_directions().savefield
        self._subsampling_directions = subsampling_directions

        # different tasks may use different samples of the same extension
        self._merged_extensions = merge_extensions(
            [
                (self._extension_cls_first, self._subsampling_first),
                (self._extension_cls_second, self._subsampling_second),
                (self._extension_cls_directions, self._subsampling_directions),
            ]
        )

        # how to access samples from the computed quantities
        merged_subsampling_first = self._merged_extensions[self._extension_cls_first]
        self._access_first = sample_output_mapping(
            self._subsampling_first, merged_subsampling_first
        )

        merged_subsampling_second = self._merged_extensions[self._extension_cls_second]
        self._access_second = sample_output_mapping(
            self._subsampling_second, merged_subsampling_second
        )

        merged_subsampling_directions = self._merged_extensions[
            self._extension_cls_directions
        ]
        self._access_directions = sample_output_mapping(
            self._subsampling_directions, merged_subsampling_directions
        )

        self._verbose = verbose

        # filled via side effects during update step computation, keys are group ids
        self._gram_evals = {}
        self._gram_evecs = {}
        self._gram_mat = {}
        self._gammas = {}
        self._lambdas = {}
        self._batch_size = {}

    def get_extensions(self) -> List[BackpropExtension]:
        """Instantiate the BackPACK extensions to compute GGN directional derivatives.

        Returns:
            BackPACK extensions, to compute directional 1ˢᵗ- and 2ⁿᵈ-order directional
            derivatives along GGN eigenvectors, that should be extracted and passed to
            the :py:class:`with backpack(...) <backpack.backpack>` context.
        """
        return [
            ext_cls(subsampling=subsampling)
            for ext_cls, subsampling in self._merged_extensions.items()
        ]

    def get_extension_hook(self, param_groups: List[Dict]) -> Callable[[Module], None]:
        """Instantiate BackPACK extension hook to compute GGN directional derivatives.

        Args:
            param_groups: Parameter groups list as required by a
                ``torch.optim.Optimizer``. Specifies the block structure: Each group
                must specify the ``'params'`` key which contains a list of the
                parameters that form a GGN block, and a ``'criterion'`` entry that
                specifies a filter function to select eigenvalues as directions along
                which to compute directional derivatives (details below).

                Examples for ``'params'``:

                - ``[{'params': list(p for p in model.parameters()}]`` uses the full
                  GGN (one block).
                - ``[{'params': [p]} for p in model.parameters()]`` uses a per-parameter
                  block-diagonal GGN approximation.

                The function specified under ``'criterion'`` is a
                ``Callable[[Tensor], List[int]]``. It receives the eigenvalues (in
                ascending order) and returns the indices of eigenvalues whose
                eigenvectors should be used as directions to evaluate directional
                derivatives. Examples:

                - ``{'criterion': lambda evals: [evals.numel() - 1]}`` discards all
                  directions except for the leading eigenvector.
                - ``{'criterion': lambda evals: list(range(evals.numel()))}`` computes
                  directional derivatives along all Gram matrix eigenvectors.

        Returns:
            BackPACK extension hook, to compute directional derivatives, that should be
            passed to the :py:class:`with backpack(...) <backpack.backpack>` context.
            The hook computes GGN directional derivatives during backpropagation and
            stores them internally (under ``self._gammas`` and ``self._lambdas``).
        """
        hook_store_batch_size = self._get_hook_store_batch_size(param_groups)

        param_computation = self.get_param_computation()
        group_hook = self.get_group_hook()
        accumulate = self.get_accumulate()

        hook = ParameterGroupsHook.from_functions(
            param_groups, param_computation, group_hook, accumulate
        )

        def extension_hook(module):
            """Extension hook executed right after BackPACK extensions during backprop.

            Chains together all the required computations.

            Args:
                module (torch.nn.Module): Layer on which the hook is executed.
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

    def get_param_computation(self):
        """Set up the ``param_computation`` function of the ``ParameterGroupsHook``.

        Returns:
            function: Function that can be bound to a ``ParameterGroupsHook`` instance.
                Performs an action on the accumulated results over parameters for a
                group.
        """
        param_computation_V_t_V = self._param_computation_V_t_V
        param_computation_V_t_g_n = self._param_computation_V_t_g_n
        param_computation_V_n_t_V = self._param_computation_V_n_t_V
        param_computation_memory_cleanup = self._param_computation_memory_cleanup

        def param_computation(self, param):
            """Compute dot products for a parameter used in directional derivatives.

            Args:
                self (ParameterGroupsHook): Group hook to which this function will be
                    bound.
                param (torch.Tensor): Parameter of a neural net.

            Returns:
                dict: Dictionary with results of the different dot products. Has key
                    ``"V_t_g_n"``.
            """
            result = {
                "V_t_V": param_computation_V_t_V(param),
                "V_t_g_n": param_computation_V_t_g_n(param),
                "V_n_t_V": param_computation_V_n_t_V(param),
            }

            param_computation_memory_cleanup(param)

            return result

        return param_computation

    def get_group_hook(self):
        """Set up the ``group_hook`` function of the ``ParameterGroupsHook``.

        Returns:
            function: Function that can be bound to a ``ParameterGroupsHook`` instance.
                Performs an action on the accumulated results over parameters for a
                group.
        """
        group_hook_directions = self._group_hook_directions
        group_hook_filter_directions = self._group_hook_filter_directions
        group_hook_gammas = self._group_hook_gammas
        group_hook_lambdas = self._group_hook_lambdas
        group_hook_memory_cleanup = self._group_hook_memory_cleanup

        def group_hook(self, accumulation, group):
            """Compute Gram space directions. Evaluate directional derivatives.

            Args:
                self (ParameterGroupsHook): Group hook to which this function will be
                    bound.
                accumulation (dict): Accumulated dot products.
                group (dict): Parameter group of a ``torch.optim.Optimizer``.
            """
            group_hook_directions(accumulation, group)
            group_hook_filter_directions(accumulation, group)
            group_hook_gammas(accumulation, group)
            group_hook_lambdas(accumulation, group)
            group_hook_memory_cleanup(accumulation, group)

        return group_hook

    def get_accumulate(self):
        """Set up the ``accumulate`` function of the ``ParameterGroupsHook``.

        Returns:
            function: Function that can be bound to a ``ParameterGroupsHook`` instance.
                Accumulates the parameter computations.
        """
        verbose = self._verbose

        def accumulate(self, existing, update):
            """Update existing results with computation result of a parameter.

            Args:
                self (ParameterGroupsHook): Group hook to which this function will be
                    bound.
                existing (dict): Dictionary containing the different accumulated scalar
                    products. Must have same keys as ``update``.
                update (dict): Dictionary containing the different scalar products for
                    a parameter.

            Returns:
                dict: Updated scalar products.

            Raises:
                ValueError: If the two inputs don't have the same keys.
                ValueError: If the two values associated to a key have different type.
                NotImplementedError: If the rule to accumulate a data type is missing.
            """
            same_keys = set(existing.keys()) == set(update.keys())
            if not same_keys:
                raise ValueError("Cached and new results have different keys.")

            for key in existing.keys():
                current, new = existing[key], update[key]

                same_type = type(current) is type(new)
                if not same_type:
                    raise ValueError(f"Value for key '{key}' have different types.")

                if isinstance(current, torch.Tensor):
                    current.add_(new)
                elif current is None:
                    pass
                else:
                    raise NotImplementedError(f"No rule for {type(current)}")

                existing[key] = current

                if verbose:
                    print(f"Accumulate group entry '{key}'")

            return existing

        return accumulate

    # parameter computations

    def _param_computation_V_t_V(self, param):
        """Perform scalar products ``V_t_V`` for a parameter.

        Args:
            param (torch.Tensor): Parameter of a neural net.

        Returns:
            torch.Tensor: Scalar products ``V_t_V``.
        """
        savefields = (self._savefield_directions, self._savefield_directions)
        subsamplings = (self._access_directions, self._access_directions)
        start_dims = (2, 2)  # only applies to GGN and GGN-MC

        tensors = self._get_subsampled_tensors(
            param, start_dims, savefields, subsamplings
        )

        if self._verbose:
            print(f"Param {id(param)}: Compute 'V_t_V'")

        return partial_contract(*tensors, start_dims)

    def _param_computation_V_t_g_n(self, param):
        """Perform scalar products ``V_t_g_n`` for a parameter.

        Args:
            param (torch.Tensor): Parameter of a neural net.

        Returns:
            torch.Tensor: Scalar products ``V_t_g_n``.
        """
        savefields = (self._savefield_directions, self._savefield_first)
        subsamplings = (self._access_directions, self._access_first)
        start_dims = (2, 1)  # only applies to (GGN or GGN-MC, BatchGrad)

        tensors = self._get_subsampled_tensors(
            param, start_dims, savefields, subsamplings
        )

        if self._verbose:
            print(f"Param {id(param)}: Compute 'V_t_g_n'")

        return partial_contract(*tensors, start_dims)

    def _param_computation_V_n_t_V(self, param):
        """Perform scalar products ``V_t_g_n`` if not fully contained in ``V_t_V``.

        Args:
            param (torch.Tensor): Parameter of a neural net.

        Returns:
            None or torch.Tensor: ``None`` if all scalar products are already computed
                through ``V_t_V``. Else returns the scalar products.
        """
        # assume same extensions for directions and derivatives
        self._different_curvatures_not_supported()

        if self._verbose:
            print(f"Param {id(param)}: Compute 'V_n_t_V'")

        # ``V_n_t_V`` already computed through ``V_t_V``
        if is_subset(self._subsampling_second, self._subsampling_directions):
            return None
        else:
            # TODO Recycle scalar products that are available from the Gram matrix
            # and only compute the missing ones
            self._warn_inefficient_subsamplings()

            # re-compute everything, easier but less efficient
            savefields = (self._savefield_second, self._savefield_directions)
            subsamplings = (self._access_second, self._access_directions)
            start_dims = (2, 2)  # only applies to (GGN or GGN-MC)

            tensors = self._get_subsampled_tensors(
                param, start_dims, savefields, subsamplings
            )

            return partial_contract(*tensors, start_dims)

    @staticmethod
    def _get_subsampled_tensors(param, start_dims, savefields, subsamplings):
        """Fetch the scalar product inputs and apply sub-sampling if necessary.

        Args:
            param (torch.Tensor): Parameter of a neural net.
            savefields ([str, str]): List containing the attribute names under which
                the processed tensors are stored inside a parameter.
            start_dims ([int, int]): List holding the dimensions at which the dot
                product contractions starts.
            subsamplings([[int], [int]]): Sub-samplings that should be applied to the
                processed tensors before the scalar product operation. The batch axis
                is automatically identified as the last before the contracted
                dimensions. An entry of ``None`` does not apply subsampling. Default:
                ``(None, None)``

        Returns:
            [torch.Tensor]: List of sub-sampled inputs for the scalar product.
        """
        tensors = []

        for start_dim, savefield, subsampling in zip(
            start_dims, savefields, subsamplings
        ):
            tensor = getattr(param, savefield)

            if subsampling is not None:
                batch_axis = start_dim - 1
                select = torch.tensor(
                    subsampling, dtype=torch.int64, device=tensor.device
                )
                tensor = tensor.index_select(batch_axis, select)

            tensors.append(tensor)

        return tensors

    def _param_computation_memory_cleanup(self, param):
        """Free buffers in a parameter that are not required anymore.

        Args:
            param (torch.Tensor): Parameter of a neural net.
        """
        savefields = {
            self._savefield_directions,
            self._savefield_first,
            self._savefield_second,
        }

        for savefield in savefields:
            delattr(param, savefield)

            if self._verbose:
                print(f"Param {id(param)}: Delete '{savefield}'")

    # group hooks

    def _group_hook_directions(self, accumulation, group):
        """Evaluate and store directions of quadratic model in the Gram space.

        Sets the following entries under the id of ``group``:

        - In ``self._gram_evals``: Eigenvalues, sorted in ascending order.
        - In ``self._gram_evecs``: Normalized eigenvectors, stacked column-wise.
        - In ``self._gram_mat``: The Gram matrix ``Vᵀ V``.

        Args:
            accumulation (dict): Dictionary with accumulated scalar products.
            group (dict): Parameter group of a ``torch.optim.Optimizer``.
        """
        group_id = id(group)
        gram_mat = accumulation["V_t_V"]

        # compensate subsampling scale
        if self._subsampling_directions is not None:
            N_dir = len(self._subsampling_directions)
            N = self._batch_size[group_id]
            gram_mat *= N / N_dir

        gram_evals, gram_evecs = stable_symeig(
            reshape_as_square(gram_mat), eigenvectors=True
        )

        # save
        self._gram_mat[group_id] = gram_mat
        self._gram_evals[group_id] = gram_evals
        self._gram_evecs[group_id] = gram_evecs

        if self._verbose:
            print(f"Group {id(group)}: Store 'gram_mat', 'gram_evals', 'gram_evecs'")

    def _group_hook_filter_directions(self, accumulation, group):
        """Filter Gram directions depending on their eigenvalues.

        Modifies the group entries in ``self._gram_evals`` and ``self._gram_evecs``.

        Args:
            accumulation (dict): Dictionary with accumulated scalar products.
            group (dict): Parameter group.
        """
        group_id = id(group)

        evals = self._gram_evals[group_id]
        evecs = self._gram_evecs[group_id]

        keep = group["criterion"](evals)

        self._gram_evals[group_id] = evals[keep]
        self._gram_evecs[group_id] = evecs[:, keep]

        if self._verbose:
            before, after = len(evals), len(keep)
            print(f"Group {id(group)}: Filter directions ({before} → {after})")

    def _group_hook_gammas(self, accumulation, group):
        """Evaluate and store first-order directional derivatives ``γ[n, d]``.

        Sets the following entries under the id of ``group``:

        - In ``self._gammas``: First-order directional derivatives.

        Args:
            accumulation (dict): Dictionary with accumulated scalar products.
            group (dict): Parameter group of a ``torch.optim.Optimizer``.
        """
        group_id = id(group)

        # L = ¹/ₙ ∑ᵢ ℓᵢ, BackPACK's BatchGrad computes ¹/ₙ ∇ℓᵢ, we have to rescale
        N = self._batch_size[group_id]

        V_t_g_n = N * accumulation["V_t_g_n"]

        # compensate subsampling scale
        if self._subsampling_directions is not None:
            N_dir = len(self._subsampling_directions)
            N = self._batch_size[group_id]
            V_t_g_n *= math.sqrt(N / N_dir)

        # NOTE Flipping the order (g_n_t_V) may be more efficient
        V_t_g_n = V_t_g_n.flatten(
            start_dim=0, end_dim=1
        )  # only applies to GGN and GGN-MC

        gammas = (
            torch.einsum("in,id->nd", V_t_g_n, self._gram_evecs[group_id])
            / self._gram_evals[group_id].sqrt()
        )

        self._gammas[group_id] = gammas

        if self._verbose:
            print(f"Group {id(group)}: Store 'gammas'")

    def _group_hook_lambdas(self, accumulation, group):
        """Evaluate and store second-order directional derivatives ``λ[n, d]``.

        Sets the following entries under the id of ``group``:

        - In ``self._lambdas``: Second-order directional derivatives.

        Args:
            accumulation (dict): Dictionary with accumulated scalar products.
            group (dict): Parameter group of a ``torch.optim.Optimizer``.
        """
        # assume same extensions for directions and derivatives
        self._different_curvatures_not_supported()

        group_id = id(group)

        gram_evals = self._gram_evals[group_id]
        gram_evecs = self._gram_evecs[group_id]
        gram_mat = self._gram_mat[group_id]

        C_dir, N_dir = gram_mat.shape[:2]
        batch_size = self._batch_size[group_id]

        # all info in Gram matrix, just slice the relevant info
        if is_subset(self._subsampling_second, self._subsampling_directions):
            V_n_T_V = gram_mat.reshape(C_dir, N_dir, C_dir * N_dir)

            idx = sample_output_mapping(
                self._subsampling_second, self._subsampling_directions
            )
            if idx is not None:
                V_n_T_V = V_n_T_V[:, idx, :]

            # compensate scale of V_n
            V_n_T_V *= math.sqrt(N_dir)

        else:
            # TODO Recycle scalar products that are available from the Gram matrix
            # and only compute the missing ones
            self._warn_inefficient_subsamplings()

            # re-compute everything, easier but less efficient
            V_n_T_V = accumulation["V_n_t_V"]

            C_second, N_second = V_n_T_V.shape[:2]
            V_n_T_V = V_n_T_V.reshape(C_second, N_second, C_dir * N_dir)

            # compensate scale of V_n
            V_n_T_V *= batch_size / math.sqrt(N_dir)

        V_n_T_V_e_d = torch.einsum("cni,id->cnd", V_n_T_V, gram_evecs)

        lambdas = (V_n_T_V_e_d**2).sum(0) / gram_evals

        self._lambdas[group_id] = lambdas

        if self._verbose:
            print(f"Group {id(group)}: Store 'lambdas'")

    def _group_hook_memory_cleanup(self, accumulation, group):
        """Free up buffers which are not required anymore for a group.

        Modifies temporary buffers.

        Args:
            accumulation (dict): Dictionary with accumulated scalar products.
            group (dict): Parameter group of a ``torch.optim.Optimizer``.
        """
        group_id = id(group)
        buffers = ["_gram_mat", "_gram_evals", "_gram_evecs", "_batch_size"]

        for b in buffers:

            if self._verbose:
                print(f"Group {group_id}: Delete '{b}'")

            getattr(self, b).pop(group_id)

    def _get_hook_store_batch_size(self, param_groups):
        """Create extension hook that stores the batch size during backpropagation.

        Args:
            param_groups (list): Parameter group list from a ``torch.optim.Optimizer``.

        Returns:
            callable: Hook function to hand into a ``with backpack(...)`` context.
                Stores the batch size under the ``self._batch_size`` dictionary for each
                group.
        """

        def hook_store_batch_size(module):
            """Store batch size internally.

            Modifies ``self._batch_size``.

            Args:
                module (torch.nn.Module): The module on which the hook is executed.
            """
            if self._batch_size == {}:
                batch_axis = 0
                batch_size = module.input0.shape[batch_axis]

                for group in param_groups:
                    group_id = id(group)

                    if self._verbose:
                        print(f"Group {group_id}: Store 'batch_size'")

                    self._batch_size[group_id] = batch_size

        return hook_store_batch_size

    def _different_curvatures_not_supported(self):
        """Raise exception if curvatures for directions and derivatives deviate.

        Raises:
            NotImplementedError: If different extensions/curvature matrices are used
                for directions and second-order directional derivatives, respectively.
        """
        if self._extension_cls_directions != self._extension_cls_second:
            raise NotImplementedError(
                "Different extensions for (directions, second) not supported."
            )

    def _warn_inefficient_subsamplings(self):
        """Issue a warning if samples for ``λ[n,k]`` are not used in the Gram matrix.

        This requires more pairwise scalar products be evaluated and makes the
        computation less efficient.
        """
        warnings.warn(
            "If subsampling_second is not a subset of subsampling_directions,"
            + " all required dot products will be re-evaluated. This is not"
            + " the most efficient, but less complex implementation."
        )
