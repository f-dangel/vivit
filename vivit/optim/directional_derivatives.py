"""Manage computations in Gram space through extension hooks."""

import math
from typing import Callable, Dict, List, Optional

import torch
from backpack.extensions import BatchGrad, SqrtGGNExact, SqrtGGNMC
from backpack.extensions.backprop_extension import BackpropExtension
from torch.nn import Module

from vivit.optim.utils import get_sqrt_ggn_extension
from vivit.utils.checks import check_subsampling_unique
from vivit.utils.gram import partial_contract, reshape_as_square
from vivit.utils.hooks import ParameterGroupsHook


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

    def get_extensions(self) -> List[BackpropExtension]:
        """Instantiate the BackPACK extensions to compute GGN directional derivatives.

        Returns:
            BackPACK extensions, to compute directional 1ˢᵗ- and 2ⁿᵈ-order directional
            derivatives along GGN eigenvectors, that should be extracted and passed to
            the :py:class:`with backpack(...) <backpack.backpack>` context.
        """
        return [
            self._extension_cls_grad(subsampling=self._subsampling_grad),
            get_sqrt_ggn_extension(
                subsampling=self._subsampling_ggn, mc_samples=self._mc_samples_ggn
            ),
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
        savefield_ggn = self._savefield_ggn
        savefield_grad = self._savefield_grad
        subsampling_ggn = self._subsampling_ggn
        subsampling_grad = self._subsampling_grad
        verbose = self._verbose

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
            V, g = DirectionalDerivativesComputation._get_subsampled_tensors(
                param,
                start_dims=(2, 1),
                savefields=(savefield_ggn, savefield_grad),
                subsamplings=(subsampling_ggn, subsampling_grad),
            )

            if verbose:
                print(f"Param {id(param)}: Compute 'V_t_V' and 'V_t_g_n'")

            result = {
                "V_t_V": partial_contract(V, V, start_dims=(2, 2)),
                "V_t_g_n": partial_contract(V, g, start_dims=(2, 1)),
            }

            DirectionalDerivativesComputation._delete_savefield(param, savefield_ggn)
            DirectionalDerivativesComputation._delete_savefield(param, savefield_grad)

            return result

        return param_computation

    def get_group_hook(self):
        """Set up the ``group_hook`` function of the ``ParameterGroupsHook``.

        Returns:
            function: Function that can be bound to a ``ParameterGroupsHook`` instance.
                Performs an action on the accumulated results over parameters for a
                group.
        """
        group_hook_lambdas = self._group_hook_lambdas
        group_hook_memory_cleanup = self._group_hook_memory_cleanup
        subsampling_ggn = self._subsampling_ggn
        verbose = self._verbose
        batch_size = self._batch_size
        gammas = self._gammas

        def group_hook(self, accumulation, group):
            """Compute Gram space directions. Evaluate directional derivatives.

            Args:
                self (ParameterGroupsHook): Group hook to which this function will be
                    bound.
                accumulation (dict): Accumulated dot products.
                group (dict): Parameter group of a ``torch.optim.Optimizer``.
            """
            group_id = id(group)
            gram_mat = accumulation["V_t_V"]

            # compensate subsampling scale
            N = batch_size[group_id]
            if subsampling_ggn is not None:
                N_dir = len(subsampling_ggn)
                gram_mat *= N / N_dir

            evals, evecs = reshape_as_square(gram_mat).symeig(eigenvectors=True)

            if verbose:
                print(
                    f"Compute {group_id}: Store 'gram_mat', 'gram_evals', 'gram_evecs'"
                )

            keep = group["criterion"](evals)

            if verbose:
                before, after = len(evals), len(keep)
                print(f"Group {group_id}: Filter directions ({before} → {after})")

            evals, evecs = evals[keep], evecs[:, keep]

            accumulation["gram_mat"] = gram_mat
            accumulation["gram_evals"] = evals
            accumulation["gram_evecs"] = evecs

            # L = ¹/ₙ ∑ᵢ ℓᵢ, BackPACK's BatchGrad computes ¹/ₙ ∇ℓᵢ, we have to rescale
            V_t_g_n = N * accumulation["V_t_g_n"]

            # compensate subsampling scale
            if subsampling_ggn is not None:
                N_dir = len(subsampling_ggn)
                V_t_g_n *= math.sqrt(N / N_dir)

            # NOTE Flipping the order (g_n_t_V) may be more efficient
            V_t_g_n = V_t_g_n.flatten(start_dim=0, end_dim=1)

            gammas[group_id] = torch.einsum("in,id->nd", V_t_g_n, evecs) / evals.sqrt()

            if verbose:
                print(f"Group {group_id}: Store 'gammas'")

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

    @staticmethod
    def _delete_savefield(param, savefield, verbose=False):
        if verbose:
            print(f"Param {id(param)}: Delete '{savefield}'")

        delattr(param, savefield)

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

    # group hooks

    def _group_hook_lambdas(self, accumulation, group):
        """Evaluate and store second-order directional derivatives ``λ[n, d]``.

        Sets the following entries under the id of ``group``:

        - In ``self._lambdas``: Second-order directional derivatives.

        Args:
            accumulation (dict): Dictionary with accumulated scalar products.
            group (dict): Parameter group of a ``torch.optim.Optimizer``.
        """
        group_id = id(group)

        gram_evals = accumulation["gram_evals"]
        gram_evecs = accumulation["gram_evecs"]
        gram_mat = accumulation["gram_mat"]

        C_dir, N_dir = gram_mat.shape[:2]

        V_n_T_V = gram_mat.reshape(C_dir, N_dir, C_dir * N_dir)

        if self._subsampling_ggn is not None:
            V_n_T_V = V_n_T_V[:, self._subsampling_ggn, :]

        # compensate scale of V_n
        V_n_T_V *= math.sqrt(N_dir)

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
        buffers = ["_batch_size"]

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
