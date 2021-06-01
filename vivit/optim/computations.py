"""Handle damped Newton step computations after and during backpropagation."""

import math
from functools import partial

from vivit.extensions import SqrtGGNExact
from vivit.optim.gram_computations import GramComputations
from vivit.utils.ggn import V_mat_prod
from vivit.utils.hooks import ParameterGroupsHook


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
            verbose (bool, optional): Turn on verbose mode. Default: ``False``.
        """
        self._gram_computation = GramComputations(
            subsampling_directions=subsampling_directions,
            subsampling_first=subsampling_first,
            subsampling_second=subsampling_second,
            extension_cls_directions=extension_cls_directions,
            extension_cls_second=extension_cls_second,
            verbose=verbose,
        )
        self._verbose = verbose

        # filled via side effects during update step computation, keys are group ids
        self._deltas = {}
        self._newton_step = {}

    def get_extensions(self, param_groups):
        """Return the instantiated BackPACK extensions required in the backward pass.

        Args:
            param_groups (list): Parameter group list from a ``torch.optim.Optimizer``.

        Returns:
            [backpack.extensions.backprop_extension.BackpropExtension]: List of
                extensions that can be handed into a ``with backpack(...)`` context.
        """
        return self._gram_computation.get_extensions(param_groups)

    def get_extension_hook(
        self,
        param_groups,
        damping,
        savefield,
        keep_gram_mat=True,
        keep_gram_evals=True,
        keep_gram_evecs=True,
        keep_gammas=True,
        keep_lambdas=True,
        keep_batch_size=True,
        keep_deltas=True,
        keep_newton_step=True,
        keep_backpack_buffers=True,
    ):
        """Return hook to be executed right after a BackPACK extension during backprop.

        Args:
            param_groups (list): Parameter group list from a ``torch.optim.Optimizer``.
            damping (vivit.optim.damping.BaseDamping): Instance for computing damping
                parameters from first- and second-order directional derivatives.
            savefield (str): Name of the attribute created in the parameters.
            keep_gram_mat (bool, optional): Keep buffers for Gram matrix under group id
                in ``self._gram_computation._gram_mat``. Default: ``True``
            keep_gram_evals (bool, optional): Keep buffers for filtered Gram matrix
                eigenvalues under group id in ``self._gram_computation._gram_evals``.
                Default: ``True``
            keep_gram_evecs (bool, optional): Keep buffers for filtered Gram matrix
                eigenvectors under group id in ``self._gram_computation._gram_evecs``.
                Default: ``True``
            keep_gammas (bool, optional): Keep buffers for first-order directional
                derivatives under group id in ``self._gram_computation._gammas``.
                Default: ``True``
            keep_lambdas (bool, optional): Keep buffers for second-order directional
                derivatives under group id in ``self._gram_computation._lambdas``.
                Default: ``True``
            keep_batch_size (bool, optional): Keep batch size for under group id
                in ``self._gram_computation._lambdas``. Default: ``True``
            keep_deltas (bool, optional): Keep directional dampings under group id in
                ``self._deltas``. Default: ``True``.
            keep_newton_step (bool, optional): Keep damped Newton step under group id
                in ``self._newton_step``. Default: ``True``.
            keep_backpack_buffers (bool, optional): Keep buffers from used BackPACK
                extensions during backpropagation. Default: ``True``.

        Returns:
            callable or None: Hook function that can be handed into a
                ``with backpack(...)`` context. ``None`` signifies no action will be
                performed.
        """
        hook_store_batch_size = self._gram_computation._get_hook_store_batch_size(
            param_groups
        )

        param_computation = self.get_param_computation()
        group_hook = self.get_group_hook(
            damping,
            savefield,
            keep_gram_mat=keep_gram_mat,
            keep_gram_evals=keep_gram_evals,
            keep_gram_evecs=keep_gram_evecs,
            keep_gammas=keep_gammas,
            keep_lambdas=keep_lambdas,
            keep_batch_size=keep_batch_size,
            keep_deltas=keep_deltas,
            keep_newton_step=keep_newton_step,
            keep_backpack_buffers=keep_backpack_buffers,
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

    def get_param_computation(self):
        """Set up the ``param_computation`` function of the ``ParameterGroupsHook``.

        Returns:
            function: Function that can be bound to a ``ParameterGroupsHook`` instance.
                Performs an action on the accumulated results over parameters for a
                group.
        """
        return self._gram_computation.get_param_computation(keep_backpack_buffers=True)

    def get_group_hook(
        self,
        damping,
        savefield,
        keep_gram_mat,
        keep_gram_evals,
        keep_gram_evecs,
        keep_gammas,
        keep_lambdas,
        keep_batch_size,
        keep_deltas,
        keep_newton_step,
        keep_backpack_buffers,
    ):
        """Set up the ``group_hook`` function of the ``ParameterGroupsHook``.

        Args:
            damping (vivit.optim.damping.BaseDamping): Instance for computing damping
                parameters from first- and second-order directional derivatives.
            savefield (str): Name of the attribute created in the parameters.
            keep_gram_mat (bool): Keep buffers for Gram matrix under group id in
                ``self._gram_computation._gram_mat``.
            keep_gram_evals (bool): Keep buffers for filtered Gram matrix
                eigenvalues under group id in ``self._gram_computation._gram_evals``.
            keep_gram_evecs (bool): Keep buffers for filtered Gram matrix
                eigenvectors under group id in ``self._gram_computation._gram_evecs``.
            keep_gammas (bool): Keep buffers for first-order directional
                derivatives under group id in ``self._gram_computation._gammas``.
            keep_lambdas (bool): Keep buffers for second-order directional
                derivatives under group id in ``self._gram_computation._lambdas``.
            keep_batch_size (bool): Keep batch size for under group id
                in ``self._gram_computation._lambdas``. Default: ``True``
            keep_deltas (bool): Keep directional dampings under group id in
                ``self._deltas``.
            keep_newton_step (bool): Keep damped Newton step under group id
                in ``self._newton_step``.
            keep_backpack_buffers (bool): Keep buffers from used BackPACK
                extensions during backpropagation.

        Returns:
            function: Function that can be bound to a ``ParameterGroupsHook`` instance.
                Performs an action on the accumulated results over parameters for a
                group.
        """
        group_hook_gram = self._gram_computation.get_group_hook(
            keep_gram_mat=True,
            keep_gram_evals=True,
            keep_gram_evecs=True,
            keep_gammas=True,
            keep_lambdas=True,
            keep_batch_size=True,
        )
        group_hook_deltas = partial(self._group_hook_deltas, damping=damping)
        group_hook_newton_step = self._group_hook_newton_step
        group_hook_load_to_params = partial(
            self._group_hook_load_to_params, savefield=savefield
        )
        group_hook_memory_cleanup = partial(
            self._group_hook_memory_cleanup,
            keep_gram_mat=keep_gram_mat,
            keep_gram_evals=keep_gram_evals,
            keep_gram_evecs=keep_gram_evecs,
            keep_gammas=keep_gammas,
            keep_lambdas=keep_lambdas,
            keep_batch_size=keep_batch_size,
            keep_deltas=keep_deltas,
            keep_newton_step=keep_newton_step,
            keep_backpack_buffers=keep_backpack_buffers,
        )

        def group_hook(self, accumulation, group):
            """Compute Newton step, load to parameter, clean up.

            Args:
                self (ParameterGroupsHook): Group hook to which this function will be
                    bound.
                accumulation (dict): Accumulated dot products.
                group (dict): Parameter group of a ``torch.optim.Optimizer``.
            """
            group_hook_gram(self, accumulation, group)
            group_hook_deltas(accumulation, group)
            group_hook_newton_step(accumulation, group)
            group_hook_load_to_params(accumulation, group)
            group_hook_memory_cleanup(accumulation, group)

        return group_hook

    def get_accumulate(self):
        """Set up the ``accumulate`` function of the ``ParameterGroupsHook``.

        Returns:
            function: Function that can be bound to a ``ParameterGroupsHook`` instance.
                Accumulates the parameter computations.
        """
        return self._gram_computation.get_accumulate()

    # group hooks

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
        keep_deltas,
        keep_newton_step,
        keep_backpack_buffers,
    ):
        """Free cached information for an optimizer group.

        Modifies temporary buffers.

        Args:
            accumulation (dict): Dictionary with accumulated information.
            group (dict): Parameter group of a ``torch.optim.Optimizer``.
            keep_gram_mat (bool): Keep buffers for Gram matrix under group id in
                ``self._gram_computation._gram_mat``.
            keep_gram_evals (bool): Keep buffers for filtered Gram matrix
                eigenvalues under group id in ``self._gram_computation._gram_evals``.
            keep_gram_evecs (bool): Keep buffers for filtered Gram matrix
                eigenvectors under group id in ``self._gram_computation._gram_evecs``.
            keep_gammas (bool): Keep buffers for first-order directional
                derivatives under group id in ``self._gram_computation._gammas``.
            keep_lambdas (bool): Keep buffers for second-order directional
                derivatives under group id in ``self._gram_computation._lambdas``.
            keep_batch_size (bool): Keep batch size for under group id
                in ``self._gram_computation._lambdas``. Default: ``True``
            keep_deltas (bool): Keep directional dampings under group id in
                ``self._deltas``.
            keep_newton_step (bool): Keep damped Newton step under group id
                in ``self._newton_step``.
            keep_backpack_buffers (bool): Keep buffers from used BackPACK
                extensions during backpropagation.
        """
        self._gram_computation._group_hook_memory_cleanup(
            accumulation,
            group,
            keep_gram_mat=keep_gram_mat,
            keep_gram_evals=keep_gram_evals,
            keep_gram_evecs=keep_gram_evecs,
            keep_gammas=keep_gammas,
            keep_lambdas=keep_lambdas,
            keep_batch_size=keep_batch_size,
        )

        savefields = {
            self._gram_computation._savefield_directions,
            self._gram_computation._savefield_first,
            self._gram_computation._savefield_second,
        }

        for param in group["params"]:
            for savefield in savefields:

                if self._verbose:
                    print(f"Param {id(param)}: Delete '{savefield}'")

                delattr(param, savefield)

        buffers = []

        if not keep_newton_step:
            buffers.append("_newton_step")

        if not keep_deltas:
            buffers.append("_deltas")

        group_id = id(group)
        for b in buffers:

            if self._verbose:
                print(f"Group {group_id}: Delete '{b}'")

            getattr(self, b).pop(group_id)

    def _group_hook_deltas(self, accumulation, group, damping):
        """Evaluate dampings for individual directions.

        Sets the following entries under the id of ``group``:

        - In ``self._deltas``: Directional dampings.

        Args:
            accumulation (dict): Dictionary with accumulated information.
            group (dict): Parameter group of a ``torch.optim.Optimizer``.
            damping (vivit.optim.damping.BaseDamping): Instance for computing damping
                parameters from first- and second-order directional derivatives.
        """
        group_id = id(group)

        gammas = self._gram_computation._gammas[group_id]
        lambdas = self._gram_computation._lambdas[group_id]

        deltas = damping(gammas, lambdas)

        self._deltas[group_id] = deltas

        if self._verbose:
            print(f"Group {id(group)}: Store '_deltas'")

    def _group_hook_newton_step(self, accumulation, group):
        """Evaluate the damped Newton update ``- ∑ᵢ ( γᵢ / (λᵢ + δᵢ)) eᵢ / ||eᵢ||``.

        Sets the following entries under the id of ``group``:

        - In ``self._newton_step``: Damped Newton step.

        Args:
            accumulation (dict): Dictionary with accumulated information.
            group (dict): Parameter group of a ``torch.optim.Optimizer``.
        """
        group_id = id(group)

        gram_evals = self._gram_computation._gram_evals[group_id]
        gram_evecs = self._gram_computation._gram_evecs[group_id]
        N_dir = (
            self._gram_computation._batch_size[group_id]
            if self._gram_computation._subsampling_directions is None
            else len(self._gram_computation._subsampling_directions)
        )
        C_dir = gram_evecs.shape[0] // N_dir
        gammas = self._gram_computation._gammas[group_id]
        lambdas = self._gram_computation._lambdas[group_id]
        deltas = self._deltas[group_id]
        V_mp = self._get_V_mat_prod(group)

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
        gram_step = gram_step.reshape(1, C_dir, N_dir)
        newton_step = [V_g.squeeze(0) for V_g in V_mp(gram_step)]

        # compensate scale of V
        N = self._gram_computation._batch_size[group_id]
        newton_step = [math.sqrt(N / N_dir) * step for step in newton_step]

        self._newton_step[group_id] = newton_step

        if self._verbose:
            print(f"Group {id(group)}: Store '_newton_step'")

    def _group_hook_load_to_params(self, accumulation, group, savefield):
        """Copy the damped Newton step to the group parameters.

        Creates a ``savefield`` attribute in each parameter of ``group``.

        Args:
            accumulation (dict): Dictionary with accumulated information.
            group (dict): Parameter group of a ``torch.optim.Optimizer``.
            savefield (str): Name of the attribute created in the parameters.
        """
        group_id = id(group)

        params = group["params"]
        newton_step = self._newton_step[group_id]

        for param, newton in zip(params, newton_step):
            self._save_to_param(param, newton, savefield)

    def _get_V_mat_prod(self, group):
        """Get multiplication with curvature matrix square root used by directions.

        Args:
            group (dict): Parameter group of a ``torch.optim.Optimizer``.

        Returns:
            function: Vectorized multiplication with curvature matrix square root ``V``.
        """
        return partial(
            V_mat_prod,
            parameters=group["params"],
            savefield=self._gram_computation._savefield_directions,
            subsampling=self._gram_computation._access_directions,
        )

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
