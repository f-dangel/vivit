"""Manage computations in Gram space through extension hooks."""

import math
import warnings
from functools import partial

import torch
from backpack.extensions import BatchGrad, SqrtGGNExact

from vivit.utils.eig import stable_symeig
from vivit.utils.gram import partial_contract, reshape_as_square
from vivit.utils.hooks import ParameterGroupsHook
from vivit.utils.subsampling import is_subset, merge_extensions, sample_output_mapping


class GramComputations:
    """Compute directions ``{(λₖ, ẽₖ)}``, slopes ``γ[n,k]`` & curvatures ``λ[n,k]``.

    Different samples may be assigned to the three steps. The computation happens
    during backpropagation via extension hooks and allows the used buffers to be
    discarded immediately afterwards.

    - ``{(λₖ, ẽₖ)}``: Directions in Gram space with associated eigenvalues.
    - ``γ[n,k]``: 1st-order directional derivative along ``eₖ`` (implied by ``ẽₖ``).
    - ``λ[n,k]``: 2nd-order directional derivative along ``eₖ`` (implied by ``ẽₖ``).
    """

    def __init__(
        self,
        subsampling_directions=None,
        subsampling_first=None,
        subsampling_second=None,
        extension_cls_directions=SqrtGGNExact,
        extension_cls_second=SqrtGGNExact,
        compute_gammas=True,
        compute_lambdas=True,
        verbose=False,
    ):
        """Store indices of samples used for each task.

        Args:
            subsampling_directions ([int] or None): Indices of samples used to compute
                Newton directions. If ``None``, all samples in the batch will be used.
            subsampling_first ([int] or None): Indices of samples used to compute first-
                order directional derivatives along the Newton directions. If ``None``,
                all samples in the batch will be used.
            subsampling_second ([int] or None): Indices of samples used to compute
                second-order directional derivatives along the Newton directions. If
                ``None``, all samples in the batch will be used.
            extension_cls_directions (backpack.backprop_extension.BackpropExtension):
                BackPACK extension class used to compute descent directions.
            extension_cls_second (backpack.backprop_extension.BackpropExtension):
                BackPACK extension class used to compute second-order directional
                derivatives.
            compute_gammas (bool, optional): Whether to compute first-order directional
                derivatives. Default: ``True``
            compute_lambdas (bool, optional): Whether to compute second-order
                directional derivatives. Default: ``True``
            verbose (bool, optional): Turn on verbose mode. Default: ``False``.
        """
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

        self._compute_gammas = compute_gammas
        self._compute_lambdas = compute_lambdas

        # safe guards if directional derivatives are not computed
        if not self._compute_gammas:
            assert subsampling_first is None
        if not self._compute_lambdas:
            assert extension_cls_second == extension_cls_directions
            assert subsampling_second == subsampling_directions

        # filled via side effects during update step computation, keys are group ids
        self._gram_evals = {}
        self._gram_evecs = {}
        self._gram_mat = {}
        self._gammas = {}
        self._lambdas = {}
        self._batch_size = {}

    def get_extensions(self, param_groups):
        """Return the instantiated BackPACK extensions required in the backward pass.

        Args:
            param_groups (list): Parameter group list from a ``torch.optim.Optimizer``.

        Returns:
            [backpack.extensions.backprop_extension.BackpropExtension]: List of
                extensions that can be handed into a ``with backpack(...)`` context.
        """
        extensions = [
            ext_cls(subsampling=subsampling)
            for ext_cls, subsampling in self._merged_extensions.items()
        ]

        if not self._compute_gammas:
            extensions = [
                ext
                for ext in extensions
                if not isinstance(ext, self._extension_cls_first)
            ]

        return extensions

    def get_extension_hook(
        self,
        param_groups,
        keep_gram_mat=True,
        keep_gram_evals=True,
        keep_gram_evecs=True,
        keep_gammas=True,
        keep_lambdas=True,
        keep_batch_size=True,
        keep_backpack_buffers=True,
    ):
        """Return hook to be executed right after a BackPACK extension during backprop.

        Args:
            param_groups (list): Parameter group list from a ``torch.optim.Optimizer``.
            keep_gram_mat (bool, optional): Keep buffers for Gram matrix under group id
                in ``self._gram_mat``. Default: ``True``
            keep_gram_evals (bool, optional): Keep buffers for filtered Gram matrix
                eigenvalues under group id in ``self._gram_evals``. Default: ``True``
            keep_gram_evecs (bool, optional): Keep buffers for filtered Gram matrix
                eigenvectors under group id in ``self._gram_evecs``. Default: ``True``
            keep_gammas (bool, optional): Keep buffers for first-order directional
                derivatives under group id in ``self._gammas``. Default: ``True``
            keep_lambdas (bool, optional): Keep buffers for second-order directional
                derivatives under group id in ``self._lambdas``. Default: ``True``
            keep_batch_size (bool, optional): Keep batch size for under group id
                in ``self._lambdas``. Default: ``True``
            keep_backpack_buffers (bool, optional): Keep buffers from used BackPACK
                extensions during backpropagation. Default: ``True``.

        Returns:
            ParameterGroupsHook: Hook that can be handed into a ``with backpack(...)``.
        """
        hook_store_batch_size = self._get_hook_store_batch_size(param_groups)

        param_computation = self.get_param_computation(keep_backpack_buffers)
        group_hook = self.get_group_hook(
            keep_gram_mat=keep_gram_mat,
            keep_gram_evals=keep_gram_evals,
            keep_gram_evecs=keep_gram_evecs,
            keep_gammas=keep_gammas,
            keep_lambdas=keep_lambdas,
            keep_batch_size=keep_batch_size,
        )
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

    def get_param_computation(self, keep_backpack_buffers):
        """Set up the ``param_computation`` function of the ``ParameterGroupsHook``.

        Args:
            keep_backpack_buffers (bool): Keep buffers from used BackPACK extensions
                during backpropagation.

        Returns:
            function: Function that can be bound to a ``ParameterGroupsHook`` instance.
                Performs an action on the accumulated results over parameters for a
                group.
        """
        param_computation_V_t_V = self._param_computation_V_t_V
        param_computation_V_t_g_n = self._param_computation_V_t_g_n
        param_computation_V_n_t_V = self._param_computation_V_n_t_V
        param_computation_memory_cleanup = partial(
            self._param_computation_memory_cleanup,
            keep_backpack_buffers=keep_backpack_buffers,
        )

        compute_gammas = self._compute_gammas
        compute_lambdas = self._compute_lambdas

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
            result = {}

            result["V_t_V"] = param_computation_V_t_V(param)

            if compute_gammas:
                result["V_t_g_n"] = param_computation_V_t_g_n(param)
            if compute_lambdas:
                result["V_n_t_V"] = param_computation_V_n_t_V(param)

            param_computation_memory_cleanup(param)

            return result

        return param_computation

    def get_group_hook(
        self,
        keep_gram_mat,
        keep_gram_evals,
        keep_gram_evecs,
        keep_gammas,
        keep_lambdas,
        keep_batch_size,
    ):
        """Set up the ``group_hook`` function of the ``ParameterGroupsHook``.

        Args:
            keep_gram_mat (bool): Keep buffers for Gram matrix under group id
                in ``self._gram_mat``.
            keep_gram_evals (bool): Keep buffers for filtered Gram matrix
                eigenvalues under group id in ``self._gram_evals``.
            keep_gram_evecs (bool): Keep buffers for filtered Gram matrix
                eigenvectors under group id in ``self._gram_evecs``.
            keep_gammas (bool): Keep buffers for first-order directional
                derivatives under group id in ``self._gammas``.
            keep_lambdas (bool): Keep buffers for second-order directional
                derivatives under group id in ``self._lambdas``.
            keep_batch_size (bool): Keep batch size for under group id
                in ``self._lambdas``.

        Returns:
            function: Function that can be bound to a ``ParameterGroupsHook`` instance.
                Performs an action on the accumulated results over parameters for a
                group.
        """
        group_hook_directions = self._group_hook_directions
        group_hook_filter_directions = self._group_hook_filter_directions
        group_hook_gammas = self._group_hook_gammas
        group_hook_lambdas = self._group_hook_lambdas
        group_hook_memory_cleanup = partial(
            self._group_hook_memory_cleanup,
            keep_gram_mat=keep_gram_mat,
            keep_gram_evals=keep_gram_evals,
            keep_gram_evecs=keep_gram_evecs,
            keep_gammas=keep_gammas,
            keep_lambdas=keep_lambdas,
            keep_batch_size=keep_batch_size,
        )

        compute_gammas = self._compute_gammas
        compute_lambdas = self._compute_lambdas

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
            if compute_gammas:
                group_hook_gammas(accumulation, group)
            if compute_lambdas:
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

    def _param_computation_memory_cleanup(self, param, keep_backpack_buffers):
        """Free buffers in a parameter that are not required anymore.

        Args:
            param (torch.Tensor): Parameter of a neural net.
            keep_backpack_buffers (bool): Keep buffers from used BackPACK
                extensions during backpropagation.
        """
        if keep_backpack_buffers:
            savefields = []
        else:
            savefields = {
                self._savefield_directions,
                self._savefield_first,
                self._savefield_second,
            }

            if not self._compute_gammas:
                savefields.remove(self._savefield_first)

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

    def _group_hook_memory_cleanup(
        self,
        accumulation,
        group,
        keep_gram_mat,
        keep_gram_evals,
        keep_gram_evecs,
        keep_gammas,
        keep_lambdas,
        keep_batch_size,
    ):
        """Free up buffers which are not required anymore for a group.

        Modifies temporary buffers.

        Args:
            accumulation (dict): Dictionary with accumulated scalar products.
            group (dict): Parameter group of a ``torch.optim.Optimizer``.
            keep_gram_mat (bool): Keep buffers for Gram matrix under group id
                in ``self._gram_mat``.
            keep_gram_evals (bool): Keep buffers for filtered Gram matrix
                eigenvalues under group id in ``self._gram_evals``.
            keep_gram_evecs (bool): Keep buffers for filtered Gram matrix
                eigenvectors under group id in ``self._gram_evecs``.
            keep_gammas (bool): Keep buffers for first-order directional
                derivatives under group id in ``self._gammas``.
            keep_lambdas (bool): Keep buffers for second-order directional
                derivatives under group id in ``self._lambdas``.
            keep_batch_size (bool): Keep batch size for under group id
                in ``self._lambdas``.
        """
        buffers = []

        if not keep_gram_mat:
            buffers.append("_gram_mat")
        if not keep_gram_evals:
            buffers.append("_gram_evals")
        if not keep_gram_evecs:
            buffers.append("_gram_evecs")
        if not keep_gammas and self._compute_gammas:
            buffers.append("_gammas")
        if not keep_lambdas and self._compute_lambdas:
            buffers.append("_lambdas")
        if not keep_batch_size:
            buffers.append("_batch_size")

        group_id = id(group)
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
