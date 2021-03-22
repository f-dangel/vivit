"""Damping policies from first- and second-order directional derivatives."""

import torch


class BaseDamping:
    """Base class for policies to determine the damping parameter.

    To create a new damping policy, the following methods need to be implemented by
    a child class:

    - ``__call__``

    """

    def __call__(self, first_derivatives, second_derivatives):
        """Determine damping parameter for each direction.

        Let ``N₁`` and ``N₂`` denote the number of samples used for computing first-
        and second-order derivatives respectively. Let ``D`` be the number of
        directions.

        Args:
            first_derivatives (torch.Tensor): 2d tensor of shape ``[N₁,D]`` with the
                gradient projections ``γ[n, d]`` of sample ``n`` along direction ``d``.
            second_derivatives (torch.Tensor): 2d tensor of shape ``[N₂, D]`` with the
                curvature projections ``λ[n, d]`` of sample ``n`` along direction ``d``.

        Returns: # noqa: DAR202
            torch.Tensor: 1d tensor of shape ``[D]`` containing the dampings ``δ[d]``
                along direction ``d``.

        Raises:
            NotImplementedError: Must be implemented by a child class.
        """
        raise NotImplementedError


class ConstantDamping(BaseDamping):
    """Constant isotropic damping."""

    def __init__(self, damping=1.0):
        """Store damping constant.

        Args:
            damping (float, optional): Damping constant. Default value uses ``1.0``.
        """
        super().__init__()

        self._damping = damping

    def __call__(self, first_derivatives, second_derivatives):
        num_directions = first_derivatives.shape[1]
        device = first_derivatives.device

        return self._damping * torch.ones(num_directions, device=device)
