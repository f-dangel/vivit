"""Assigning mini-batch samples in a mini-batch to computations."""

import torch
from backpack.extensions import BatchGrad

from lowrank.extensions import SqrtGGNExact
from lowrank.utils.eig import symeig
from lowrank.utils.gram import (
    compute_gram_mat,
    get_letters,
    sqrt_gram_mat_prod,
    sqrt_gram_t_mat_prod,
)


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
        extension_cls_first=BatchGrad,
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
            extension_cls_first (backpack.backprop_extension.BackpropExtension):
                BackPACK extension class used to compute first-order directional
                derivatives.
            extension_cls_second (backpack.backprop_extension.BackpropExtension):
                BackPACK extension class used to compute second-order directional
                derivatives.
        """
        self._subsampling_directions = subsampling_directions
        self._subsampling_first = subsampling_first
        self._subsampling_second = subsampling_second
        self._extension_cls_directions = extension_cls_directions
        self._extension_cls_first = extension_cls_first
        self._extension_cls_second = extension_cls_second

    def get_extension_hook(self, param_groups):
        """Return hook to be executed right after a BackPACK extension during backprop.

        Args:
            param_groups (list): Parameter group list from a ``torch.optim.Optimizer``.

        Returns:
            callable or None: Hook function that can be handed into a
                ``with backpack(...)`` context. ``None`` signifies no action will be
                performed.
        """
        self._no_groups(param_groups)
        return None

    def get_extensions(self, param_groups):
        """Return the instantiated BackPACK extensions required in the backward pass.

        Args:
            param_groups (list): Parameter group list from a ``torch.optim.Optimizer``.

        Returns:
            [backpack.extensions.backprop_extension.BackpropExtension]: List of
                extensions that can be handed into a ``with backpack(...)`` context.
        """
        self._no_groups(param_groups)
        # NOTE Special care has to be taken if the same curvatures are used.
        # Samples need to be properly merged. These checks avoid this situation
        # TODO Allow subsampling. Requires logic to merge subsamplings.
        self._no_subsampling()
        # TODO Allow different curvatures. Requires logic to set up returned list.
        self._same_second_order()

        second_order_extension_cls = tuple(
            {
                self._extension_cls_directions,
                self._extension_cls_second,
            }
        )[0]
        second_order_subsampling = tuple(
            {
                self._subsampling_directions,
                self._subsampling_second,
            }
        )[0]

        return [
            self._extension_cls_first(subsampling=self._subsampling_first),
            second_order_extension_cls(subsampling=second_order_subsampling),
        ]

    def _no_subsampling(self):
        """Raise exception if subsampling is enabled.

        Raises:
            ValueError: If subsampling is enabled.
        """
        for subsampling in [
            self._subsampling_directions,
            self._subsampling_first,
            self._subsampling_second,
        ]:
            if subsampling is not None:
                raise ValueError("Subsampling is not supported.")

    def _same_second_order(self):
        """Raise exception if different curvature second-order is used.

        Raises:
            ValueError: If different curvatures are used.
        """
        if self._extension_cls_second != self._extension_cls_directions:
            raise ValueError("Different second-order extensions are not supported.")

    def _no_groups(self, param_groups):
        """Raise exception if multiple parameter groups are used.

        Args:
            param_groups (dict): Parameter group dictionary from a
                ``torch.optim.Optimizer``.

        Raises:
            ValueError: If multiple groups are used.
        """
        allowed = 1
        groups = len(param_groups)

        if groups != allowed:
            raise ValueError(f"{allowed} parameter group(s) allowed. Got {groups}.")

    def compute_step(self, group, damping, savefield):
        """Compute the damped Newton update and save it as attribute in each parameter.

        Args:
            group (dict): Entry of a ``torch.optim.Optimizer``'s parameter group.
            damping (lowrank.optim.damping.BaseDamping): Instance for computing damping
                parameters from first- and second-order directional derivatives.
            savefield (str): Attribute name under which the step will be saved in a
                parameter.
        """
        params = group["params"]
        start_dim = 2

        # directions (λ[d], ẽ[d])
        direction_savefield = self._extension_cls_directions().savefield
        V_T_V = compute_gram_mat(params, direction_savefield, start_dim, flatten=False)
        C, N = V_T_V.shape[:2]
        V_T_V = V_T_V.reshape(C * N, C * N)
        gram_evals, gram_evecs = symeig(V_T_V, eigenvectors=True, atol=1e-5)

        # first-order derivatives γ[n, d]
        first_savefield = self._extension_cls_first().savefield
        g_n = [N * getattr(p, first_savefield) for p in params]
        V_T_g_n = sqrt_gram_t_mat_prod(g_n, params, direction_savefield, start_dim)
        gammas = torch.einsum("ni,id->nd", V_T_g_n, gram_evecs) * gram_evals

        batch_axis = 0
        gammas_mean = gammas.mean(batch_axis)

        # second-order derivatives λ[n, d]
        V_n_T_V = V_T_V.reshape(C, N, C * N)
        V_n_T_V_e_d = torch.einsum("cni,id->cnd", V_n_T_V, gram_evecs)
        lambdas = (V_n_T_V_e_d ** 2).sum(0) / gram_evals

        # TODO Choose lambda: Either from directions, or second derivatives
        lambdas_mean = gram_evals
        # batch_axis = 0
        # lambdas_mean = lambdas.mean(batch_axis)

        # dampings δ[d]
        deltas = damping(gammas, lambdas)

        # TODO Expanding directions is expensive. but likely difficult to circumvent
        evecs = sqrt_gram_mat_prod(gram_evecs, params, direction_savefield, start_dim)
        # normalize
        evecs = [evec / gram_evals.sqrt() for evec in evecs]

        # update
        for p, p_directions in zip(params, evecs):
            p_step = self._damped_newton_step(
                gammas_mean, lambdas_mean, deltas, p_directions
            )
            self._save_step(p, p_step, savefield)

    @staticmethod
    def _save_step(param, step, savefield):
        """Save ``step`` in ``param`` under ``savefield``.

        Args:
            param (torch.nn.Parameter): Parameter to which ``step`` is attached.
            step (torch.Tensor): Saved quantity.
            savefield (str): Name of the attribute to save ``step`` in.

        Raises:
            ValueError: If the attribute field is already occupied.

        """
        if hasattr(param, savefield):
            raise ValueError(f"Savefield {savefield} already exists.")
        else:
            setattr(param, savefield, step)

    @staticmethod
    def _damped_newton_step(gammas, lambdas, deltas, directions):
        """Compute the damped Newton update ``- ∑ᵢ ( γᵢ / (λᵢ + δᵢ)) eᵢ``.

        The sum runs over all directions ``i``. ``γᵢ`` is the expected first-order
        derivative along direction ``eᵢ``. ``λᵢ`` is the expected second-order
        derivative along direction ``eᵢ``. Let ``D`` be the number of directions.

        Args:
            gammas (torch.Tensor): 2d tensor of shape ``[D]`` with the expected
                slope ``γᵢ`` along direction ``eᵢ``.
            lambdas (torch.Tensor): 2d tensor of shape ``[D]`` with the expected
                curvature ``λᵢ`` along direction ``eᵢ``.
            deltas (torch.Tensor): 1d tensor of shape ``[D]`` containing the dampings
                ``δᵢ`` along direction ``eᵢ``.
            directions (torch.Tensor): Tensor of shape ``[*, D]`` where ``*`` is the
                associated parameter's shape. Contains directions ``eᵢ``.

        Returns:
            torch.Tensor: Damped Newton step of same shape as the associated parameter.
        """
        letters = get_letters(directions.dim())
        equation = f"{letters[0]},{letters[1:]}{letters[0]}->{letters[1:]}"

        return -torch.einsum(equation, gammas / (lambdas + deltas), directions)
