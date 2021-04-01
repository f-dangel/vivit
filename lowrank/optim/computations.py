"""Assigning mini-batch samples in a mini-batch to computations."""

import math
import warnings
from functools import partial

import torch
from backpack.extensions import BatchGrad

from lowrank.extensions import SqrtGGNExact
from lowrank.utils.ggn import V1_t_V2, V_mat_prod, V_t_mat_prod, V_t_V
from lowrank.utils.gram import reshape_as_square
from lowrank.utils.subsampling import is_subset, merge_extensions, sample_output_mapping


class BaseComputations:
    """Base class for assigning mini-batch samples in a mini-batch to computations.

    The algorithms rely on three fundamental steps, to which samples may be assigned:

    - Computing the Newton directions ``{e[d]}``.
    - Computing the first-order derivatives ``{γ[n,d]}`` along the directions.
    - Computing the second-order derivatives ``{λ[n,d]}`` along the directions.

    The three mini-batch subsets used for each task need not be disjoint.
    """

    def __init__(
        self,
        subsampling_directions=None,
        subsampling_first=None,
        subsampling_second=None,
        extension_cls_directions=SqrtGGNExact,
        extension_cls_second=SqrtGGNExact,
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

        # filled via side effects during update step computation, keys are group ids
        self._gram_evals = {}
        self._gram_evecs = {}
        self._gram_mat = {}
        self._V_t_mat_prod = {}
        self._V_mat_prod = {}
        self._gammas = {}
        self._lambdas = {}
        self._deltas = {}
        self._newton_step = {}
        self._batch_size = {}

    def get_extension_hook(self, param_groups):
        """Return hook to be executed right after a BackPACK extension during backprop.

        Args:
            param_groups (list): Parameter group list from a ``torch.optim.Optimizer``.

        Returns:
            callable or None: Hook function that can be handed into a
                ``with backpack(...)`` context. ``None`` signifies no action will be
                performed.
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
                    self._batch_size[group_id] = batch_size

        return hook_store_batch_size

    def get_extensions(self, param_groups):
        """Return the instantiated BackPACK extensions required in the backward pass.

        Args:
            param_groups (list): Parameter group list from a ``torch.optim.Optimizer``.

        Returns:
            [backpack.extensions.backprop_extension.BackpropExtension]: List of
                extensions that can be handed into a ``with backpack(...)`` context.
        """
        return [
            ext_cls(subsampling=subsampling)
            for ext_cls, subsampling in self._merged_extensions.items()
        ]

    def compute_step(self, group, damping, savefield):
        """Compute damped Newton step and save it in attribute in each group parameter.

        Args:
            group (dict): Entry of a ``torch.optim.Optimizer``'s parameter group.
            damping (lowrank.optim.damping.BaseDamping): Instance for computing damping
                parameters from first- and second-order directional derivatives.
            savefield (str): Attribute name under which the step will be saved in a
                parameter.
        """
        # fill values for directions (λ[d], ẽ[d])
        self._eval_directions(group)

        # filter directions (λ[d], ẽ[d])
        self._filter_directions(group)

        # first-order derivatives γ[n, d]
        self._eval_gammas(group)

        # second-order derivatives λ[n, d]
        self._eval_lambdas(group)

        # dampings δ[d]
        self._eval_deltas(group, damping)

        # Newton step - ∑ᵢ γᵢ / (δᵢ + λᵢ) eᵢ / ||eᵢ||
        self._eval_newton_step(group)
        self._load_newton_step_to_params(group, savefield)

        # clean up
        self._remove_from_temp_buffers(group)

    def _eval_directions(self, group):
        """Evaluate and store information about the quadratic model's direction.

        Sets the following entries under the id of ``group``:

        - In ``self._gram_evals``: Eigenvalues, sorted in ascending order.
        - In ``self._gram_evecs``: Normalized eigenvectors, stacked column-wise.
        - In ``self._gram_mat``: The Gram matrix ``Vᵀ V``.
        - In ``self._V_t_mat_prod``: Vectorized multiplication with ``Vᵀ``.
        - In ``self._V_mat_prod``: Vectorized multiplication with ``V``.

        Args:
            group (dict): Parameter group of a ``torch.optim.Optimizer``.
        """
        group_id = id(group)

        params = group["params"]
        savefield = self._savefield_directions
        subsampling = self._access_directions

        gram_mat = V_t_V(params, savefield, subsampling=subsampling)

        # compensate subsampling scale
        if self._subsampling_directions is not None:
            N_dir = len(self._subsampling_directions)
            N = self._batch_size[group_id]
            gram_mat *= N / N_dir

        gram_evals, gram_evecs = reshape_as_square(gram_mat).symeig(eigenvectors=True)

        V_t_mp = partial(
            V_t_mat_prod,
            parameters=params,
            savefield=savefield,
            subsampling=subsampling,
        )
        V_mp = partial(
            V_mat_prod, parameters=params, savefield=savefield, subsampling=subsampling
        )

        # save
        self._gram_mat[group_id] = gram_mat
        self._gram_evals[group_id] = gram_evals
        self._gram_evecs[group_id] = gram_evecs
        self._V_t_mat_prod[group_id] = V_t_mp
        self._V_mat_prod[group_id] = V_mp

    def _filter_directions(self, group):
        """Filter directions depending on their eigenvalues.

        Modifies the group entries in ``self._gram_evals`` and ``self._gram_evecs``.

        Args:
            group (dict): Parameter group of a ``torch.optim.Optimizer``.
        """
        group_id = id(group)

        evals = self._gram_evals[group_id]
        evecs = self._gram_evecs[group_id]

        keep = group["criterion"](evals)

        self._gram_evals[group_id] = evals[keep]
        self._gram_evecs[group_id] = evecs[:, keep]

    def _eval_gammas(self, group):
        """Evaluate and store first-order directional derivatives ``γ[n, d]``.

        Must be called after ``self._eval_directions``.

        Sets the following entries under the id of ``group``:

        - In ``self._gammas``: First-order directional derivatives.

        Args:
            group (dict): Parameter group of a ``torch.optim.Optimizer``.
        """
        group_id = id(group)

        g_n = [getattr(p, self._savefield_first) for p in group["params"]]

        # apply subsampling
        if self._access_first is not None:
            print(self._access_first)
            g_n = [g[self._access_first] for g in g_n]

        # L = ¹/ₙ ∑ᵢ ℓᵢ, BackPACK's BatchGrad computes ¹/ₙ ∇ℓᵢ, we have to rescale
        N = self._batch_size[group_id]
        g_n = [N * g for g in g_n]

        V_t_g_n = self._V_t_mat_prod[group_id](g_n, flatten=True)

        # compensate subsampling scale
        if self._subsampling_directions is not None:
            N_dir = len(self._subsampling_directions)
            N = self._batch_size[group_id]
            V_t_g_n *= math.sqrt(N / N_dir)

        gammas = (
            torch.einsum("ni,id->nd", V_t_g_n, self._gram_evecs[group_id])
            / self._gram_evals[group_id].sqrt()
        )

        self._gammas[group_id] = gammas

    def _eval_lambdas(self, group):
        """Evaluate and store second-order directional derivatives ``λ[n, d]``.

        Must be called after ``self._eval_directions``.

        Sets the following entries under the id of ``group``:

        - In ``self._lambdas``: Second-order directional derivatives.

        Args:
            group (dict): Parameter group of a ``torch.optim.Optimizer``.

        Raises:
            NotImplementedError: If the computation of ``λ[n, d]`` would require
                more scalar products than those contained in the Gram matrix.
        """
        group_id = id(group)

        gram_evals = self._gram_evals[group_id]
        gram_evecs = self._gram_evecs[group_id]
        gram_mat = self._gram_mat[group_id]

        C_dir, N_dir = gram_mat.shape[:2]
        batch_size = self._batch_size[group_id]

        if self._extension_cls_directions == self._extension_cls_second:
            # all info computed, just slice the relevant info
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
                warnings.warn(
                    "If subsampling_second is not a subset of subsampling_directions,"
                    + " all required dot products will be re-evaluated. This is not"
                    + " the most efficient, but less complex implementation."
                )

                # re-compute everything, easier but less efficient
                V_n_T_V = V1_t_V2(
                    group["params"],
                    self._savefield_second,
                    self._savefield_directions,
                    subsampling1=self._access_second,
                    subsampling2=self._access_directions,
                )
                C_second, N_second = V_n_T_V.shape[:2]
                V_n_T_V = V_n_T_V.reshape(C_second, N_second, C_dir * N_dir)

                # compensate scale of V_n
                V_n_T_V *= batch_size / math.sqrt(N_dir)

        else:
            raise NotImplementedError(
                "Different extensions for (directions, second) not supported."
            )

        V_n_T_V_e_d = torch.einsum("cni,id->cnd", V_n_T_V, gram_evecs)

        lambdas = (V_n_T_V_e_d ** 2).sum(0) / gram_evals

        self._lambdas[group_id] = lambdas

    def _eval_deltas(self, group, damping):
        """Evaluate dampings for individual directions.

        Must be called after ``self._eval_gammas`` and ``self.eval_lambdas``.

        Sets the following entries under the id of ``group``:

        - In ``self._deltas``: Directional dampings.

        Args:
            group (dict): Parameter group of a ``torch.optim.Optimizer``.
            damping (lowrank.optim.damping.BaseDamping): Policy for selecting
                dampings along a direction from first- and second- order directional
                derivatives.
        """
        group_id = id(group)

        gammas = self._gammas[group_id]
        lambdas = self._lambdas[group_id]

        deltas = damping(gammas, lambdas)

        self._deltas[group_id] = deltas

    def _eval_newton_step(self, group):
        """Evaluate the damped Newton update ``- ∑ᵢ ( γᵢ / (λᵢ + δᵢ)) eᵢ / ||eᵢ||``.

        Must be called after ``self._eval_directions``, ``self._eval_gammas``,
        ``self.eval_lambdas``, and ``self._eval_deltas``.

        Sets the following entries under the id of ``group``:

        - In ``self._newton_step``: Damped Newton step.

        Args:
            group (dict): Parameter group of a ``torch.optim.Optimizer``.
        """
        group_id = id(group)

        gram_evals = self._gram_evals[group_id]
        gram_evecs = self._gram_evecs[group_id]
        gram_mat = self._gram_mat[group_id]
        gammas = self._gammas[group_id]
        lambdas = self._lambdas[group_id]
        deltas = self._deltas[group_id]
        V_mp = self._V_mat_prod[group_id]

        batch_axis = 0
        gammas_mean = gammas.mean(batch_axis)

        # TODO Choose lambda: Either from directions, or second derivatives
        use_lambda_from_directions = False
        if use_lambda_from_directions:
            lambdas_mean = gram_evals
        else:
            lambdas_mean = lambdas.mean(batch_axis)

        """
        Don't expand directions in parameter space. Instead, use

        ``eᵢ / ||eᵢ|| = V ẽᵢ / √λᵢ``

        to perform the summation over ``i`` in the Gram matrix space,

        ``- ∑ᵢ (γᵢ / (λᵢ + δᵢ)) eᵢ / ||eᵢ|| = V [∑ᵢ (γᵢ / (λᵢ + δᵢ)) ẽᵢ]``.
        """
        gram_step = (
            -gammas_mean / (lambdas_mean + deltas) / gram_evals.sqrt() * gram_evecs
        ).sum(1)
        C, N_dir = gram_mat.shape[:2]
        gram_step = gram_step.reshape(1, C, N_dir)
        newton_step = [V_g.squeeze(0) for V_g in V_mp(gram_step)]

        # compensate scale of V
        N = self._batch_size[group_id]
        newton_step = [math.sqrt(N / N_dir) * step for step in newton_step]

        self._newton_step[group_id] = newton_step

    def _load_newton_step_to_params(self, group, savefield):
        """Copy the damped Newton step to the group parameters.

        Must be called after ``self._eval_newton``.

        Creates a ``savefield`` attribute in each parameter of ``group``.

        Args:
            group (dict): Parameter group of a ``torch.optim.Optimizer``.
            savefield (str): Name of the attribute created in the parameters.
        """
        group_id = id(group)

        params = group["params"]
        newton_step = self._newton_step[group_id]

        for param, newton in zip(params, newton_step):
            self._save_to_param(param, newton, savefield)

    @staticmethod
    def _save_to_param(param, value, savefield):
        """Save ``value`` in ``param`` under ``savefield``.

        Args:
            param (torch.nn.Parameter): Parameter to which ``value`` is attached.
            value (any): Saved quantity.
            savefield (str): Name of the attribute to save ``value`` in.

        Raises:
            ValueError: If the attribute field is already occupied.
        """
        if hasattr(param, savefield):
            raise ValueError(f"Savefield {savefield} already exists.")
        else:
            setattr(param, savefield, value)

    def _remove_from_temp_buffers(self, group):
        """Free cached information for an optimizer group.

        Modifies all temporary buffers.

        Args:
            group (dict): Parameter group of a ``torch.optim.Optimizer``.
        """
        group_id = id(group)

        for buffer in [
            self._gram_evals,
            self._gram_evecs,
            self._gram_mat,
            self._V_t_mat_prod,
            self._V_mat_prod,
            self._gammas,
            self._lambdas,
            self._deltas,
            self._newton_step,
            self._batch_size,
        ]:
            buffer.pop(group_id)
