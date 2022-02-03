"""Damping policies from first- and second-order directional derivatives."""

from typing import Dict, Tuple

import torch
from torch import Tensor


class _DirectionalCoefficients:
    """Base class defining the interface for computing Newton step coefficients."""

    def compute_coefficients(
        self, first_derivatives: Tensor, second_derivatives: Tensor
    ) -> Tensor:
        """Compute the Newton step coefficients.

        Let ``N₁`` and ``N₂`` denote the number of samples used for computing first-
        and second-order derivatives respectively. Let ``D`` be the number of
        directions.

        Args:
            first_derivatives: 2d tensor of shape ``[N₁,D]`` with the
                gradient projections ``γ[n, d]`` of sample ``n`` along direction ``d``.
            second_derivatives: 2d tensor of shape ``[N₂, D]`` with the
                curvature projections ``λ[n, d]`` of sample ``n`` along direction ``d``.

        Returns: # noqa: DAR202
            1d tensor of shape ``[D]`` with coefficients ``c[d]`` along direction ``d``.

        Raises:
            NotImplementedError: Must be implemented by a child class.
        """
        raise NotImplementedError


class _Damping(_DirectionalCoefficients):
    """Base class for policies to determine the damping parameter.

    To create a new damping policy, the following methods need to be implemented by
    a child class:

    - ``__call__``

    """

    def __init__(self, save_history: bool = False):
        """Initialize damping, enable saving of previously computed values.

        Args:
            save_history: Whether to store the computed dampings. Default: ``False``.
                Only use this option if you need access to the damping values (e.g.
                for logging).
        """
        self._save_history = save_history
        self._history: Dict[Tuple[int, int], Tensor] = {}

    def compute_coefficients(
        self, first_derivatives: Tensor, second_derivatives: Tensor
    ) -> Tensor:
        """Compute Newton step coefficients ``cₖ = - γₖ / (λₖ + δₖ)``.

        Let ``N₁`` and ``N₂`` denote the number of samples used for computing first-
        and second-order derivatives respectively. Let ``D`` be the number of
        directions.

        Args:
            first_derivatives: 2d tensor of shape ``[N₁,D]`` with the
                gradient projections ``γ[n, d]`` of sample ``n`` along direction ``d``.
            second_derivatives: 2d tensor of shape ``[N₂, D]`` with the
                curvature projections ``λ[n, d]`` of sample ``n`` along direction ``d``.

        Returns:
            1d tensor of shape ``[D]`` with coefficients ``c[d]`` along direction ``d``.
        """
        batch_axis = 0
        gammas_mean = first_derivatives.mean(batch_axis)
        lambdas_mean = second_derivatives.mean(batch_axis)

        deltas = self.__call__(first_derivatives, second_derivatives)

        return -gammas_mean / (lambdas_mean + deltas)

    def __call__(self, first_derivatives: Tensor, second_derivatives: Tensor) -> Tensor:
        """Determine damping parameter for each direction.

        Let ``N₁`` and ``N₂`` denote the number of samples used for computing first-
        and second-order derivatives respectively. Let ``D`` be the number of
        directions.

        Args:
            first_derivatives: 2d tensor of shape ``[N₁,D]`` with the
                gradient projections ``γ[n, d]`` of sample ``n`` along direction ``d``.
            second_derivatives: 2d tensor of shape ``[N₂, D]`` with the
                curvature projections ``λ[n, d]`` of sample ``n`` along direction ``d``.

        Returns:
            1d tensor of shape ``[D]`` with dampings ``δ[d]`` along direction ``d``.
        """
        damping = self.compute_damping(first_derivatives, second_derivatives)

        if self._save_history:
            key = (id(first_derivatives), id(second_derivatives))
            self._history[key] = damping

        return damping

    def get_from_history(
        self, first_derivatives: Tensor, second_derivatives: Tensor, pop: bool = False
    ) -> Tensor:
        """Load previously computed damping values from history.

        Args:
            first_derivatives: First input used for damping in ``compute_damping``.
            second_derivatives: Second input used for damping in ``compute_damping``.
            pop: Whether to pop the returned value from the internal saved ones.
                Default: ``False``.

        Returns:
            Damping value from history.
        """
        key = (id(first_derivatives), id(second_derivatives))

        return self._history.pop(key) if pop else self._history[key]

    def compute_damping(
        self, first_derivatives: Tensor, second_derivatives: Tensor
    ) -> Tensor:
        """Compute the damping for each direction.

        Let ``N₁`` and ``N₂`` denote the number of samples used for computing first-
        and second-order derivatives respectively. Let ``D`` be the number of
        directions.

        Args:
            first_derivatives: 2d tensor of shape ``[N₁,D]`` with the gradient
                projections ``γ[n, d]`` of sample ``n`` along direction ``d``.
            second_derivatives: 2d tensor of shape ``[N₂, D]`` with the curvature
                projections ``λ[n, d]`` of sample ``n`` along direction ``d``.

        Returns: # noqa: DAR202
            1d tensor of shape ``[D]`` with dampings ``δ[d]`` along direction ``d``.

        Raises:
            NotImplementedError: Must be implemented by a child class.
        """
        raise NotImplementedError


class ConstantDamping(_Damping):
    """Constant isotropic damping."""

    def __init__(self, damping: float = 1.0, save_history: bool = False):
        """Store damping constant.

        Args:
            damping: Damping constant. Default value uses ``1.0``.
            save_history: Whether to store the computed dampings. Default: ``False``.
                Only use this option if you need access to the damping values (e.g.
                for logging).
        """
        super().__init__(save_history=save_history)

        self._damping = damping

    def compute_damping(
        self, first_derivatives: Tensor, second_derivatives: Tensor
    ) -> Tensor:
        num_directions = first_derivatives.shape[1]
        device = first_derivatives.device

        return self._damping * torch.ones(num_directions, device=device)


class BootstrapDamping(_Damping):
    """Adaptive damping, uses Bootstrap to generate gain samples."""

    DEFAULT_DAMPING_GRID = torch.logspace(-3, 2, 100)

    def __init__(
        self,
        damping_grid: Tensor = None,
        num_resamples: int = 100,
        percentile: float = 95.0,
        save_history: bool = False,
    ):
        """Store ``damping_grid``, ``num_resamples`` and ``percentile``.

        Args:
            damping_grid: The Bootstrap generates gain samples for all damping values
                in ``damping_grid``. Default is a log-equidistant grid between
                ``1e-3`` and ``1e2``.
            num_resamples: Number of gain samples that are generated using the
                Bootstrap. The default value is ``100``.
            percentile: Policy for delta finds a curve (among the Bootstrap gain
                samples), such that ``percentile`` percent of the gain samples lie
                above it. The default value is ``95.0``.
            save_history: Whether to store the computed dampings. Default: ``False``.
                Only use this option if you need access to the damping values (e.g.
                for logging).
        """
        super().__init__(save_history=save_history)

        self._damping_grid = (
            damping_grid if damping_grid is not None else self.DEFAULT_DAMPING_GRID
        )
        self._num_resamples = num_resamples
        self._percentile = percentile

    def _resample(self, sample):
        """Create resample of ``sample``.

        Args:
            sample (torch.Tensor): 1d ``torch.Tensor``

        Returns:
            torch.Tensor: A 1d ``torch.Tensor`` whose size is the same as ``sample``
                and whose entries are sampled with replacement from ``sample``.
        """

        N = len(sample)
        return sample[torch.randint(low=0, high=N, size=(N,))]

    def _delta_policy(self, gains):
        """Compute damping based on gains generated by the Bootstrap.

        Args:
            gains (torch.Tensor): 2d ``torch.Tensor`` of shape ``[num_resamples,
                num_dampings]``, i.e. each row corresponds to one gain resample,
                where the gain is evaluated for all dampings in ``damping_grid``.

        Returns:
            float or float("inf"): The "optimal" damping. In case no reasonable
                damping is found, it will return ``float("inf")``.
        """

        # Compute gain percentile
        q = 1 - self._percentile / 100.0
        gain_perc = torch.quantile(gains, q, dim=0)

        # Filter for positive entries in gain_perc
        ge_zero = gain_perc >= 0
        if torch.any(ge_zero):
            damping_grid_filtered = self._damping_grid[ge_zero]
            gain_perc_filtered = gain_perc[ge_zero]
            max_idx = torch.argmax(gain_perc_filtered)
            return damping_grid_filtered[max_idx]
        else:
            return float("inf")

    def compute_damping(
        self, first_derivatives: Tensor, second_derivatives: Tensor
    ) -> Tensor:
        """Determine damping parameter for each direction.

        Let ``N₁`` and ``N₂`` denote the number of samples used for computing first-
        and second-order derivatives respectively. Let ``D`` be the number of
        directions.

        Args:
            first_derivatives: 2d tensor of shape ``[N₁,D]`` with the
                gradient projections ``γ[n, d]`` of sample ``n`` along direction ``d``.
            second_derivatives: 2d tensor of shape ``[N₂, D]`` with the
                curvature projections ``λ[n, d]`` of sample ``n`` along direction ``d``.

        Returns:
            1d tensor of shape ``[D]`` with dampings ``δ[d]`` along direction ``d``.
        """
        D = first_derivatives.shape[1]
        num_dampings = len(self._damping_grid)
        device = first_derivatives.device

        self._damping_grid = self._damping_grid.to(device)

        # Vector for dampings for each direction
        dampings = torch.zeros(D, device=device)

        for D_idx in range(D):

            # Extract first and second derivatives for current direction
            first = first_derivatives[:, D_idx]
            second = second_derivatives[:, D_idx]

            # Create gain samples for every delta in self._damping_grid
            gains = torch.zeros(self._num_resamples, num_dampings).to(device)

            for resample_idx in range(self._num_resamples):

                # Resample gamma_hat and lambda_hat
                gam_hat_re = torch.mean(self._resample(first))
                lam_hat_re = torch.mean(self._resample(second))

                # Resample tau_hat
                tau_hat_re = -torch.mean(self._resample(first)) / (
                    torch.mean(self._resample(second)) + self._damping_grid
                )

                # Compute gain and store sample in gains
                gain = -gam_hat_re * tau_hat_re - 0.5 * lam_hat_re * tau_hat_re**2
                gains[resample_idx, :] = gain

            # Compute damping based on gains
            dampings[D_idx] = self._delta_policy(gains)

        return dampings


class BootstrapDamping2(_Damping):
    """Adaptive damping, uses Bootstrap to generate gain samples.

    This version differs from ``BootstrapDamping`` with regard to two aspects:

    - First, we don't resample the Newton step ``tau_hat_re``, i.e. we assume this step
      to be fixed and we only evaluate the corresponding gain for thsi step in different
      (resampled) metrics.
    - Second, when resampling gamma and lambda, we use the same resampling indices for
      both vectors, because gamma and lambda may be correlated and we loose this
      correlation, when we resample them independently. That means: We assume, that
      the gammas and lambdas are evaluated on the same samples.
    """

    DEFAULT_DAMPING_GRID = torch.logspace(-3, 2, 100)

    def __init__(
        self,
        damping_grid: Tensor = None,
        num_resamples: int = 100,
        percentile: float = 95.0,
        save_history: bool = False,
    ):
        """Store ``damping_grid``, ``num_resamples`` and ``percentile``.

        Args:
            damping_grid: The Bootstrap generates gain samples
                for all damping values in ``damping_grid``. Default is a log-
                equidistant grid between ``1e-3`` and ``1e2``.
            num_resamples: This is the number of gain samples that are
                generated using the Bootstrap. The default value is ``100``.
            percentile: The policy for delta finds a curve (among the
                Bootstrap gain samples), such that ``percentile`` percent of the
                gain samples lie above it. The default value is ``95.0``.
            save_history: Whether to store the computed dampings. Default: ``False``.
                Only use this option if you need access to the damping values (e.g.
                for logging).
        """
        super().__init__(save_history=save_history)

        self._damping_grid = (
            damping_grid if damping_grid is not None else self.DEFAULT_DAMPING_GRID
        )
        self._num_resamples = num_resamples
        self._percentile = percentile

    def _resample(self, sample):
        """Create resample of ``sample``.

        Args:
            sample (torch.Tensor): 1d ``torch.Tensor``

        Returns:
            torch.Tensor: A 1d ``torch.Tensor`` whose size is the same as ``sample``
                and whose entries are sampled with replacement from ``sample``.
        """
        N = len(sample)
        return sample[torch.randint(low=0, high=N, size=(N,))]

    def _delta_policy(self, gains):
        """Compute damping based on gains generated by the Bootstrap.

        Args:
            gains (torch.Tensor): 2d ``torch.Tensor`` of shape ``[num_resamples,
                num_dampings]``, i.e. each row corresponds to one gain resample,
                where the gain is evaluated for all dampings in ``damping_grid``.

        Returns:
            float or float("inf"): The "optimal" damping. In case no reasonable
                damping is found, it will return ``float("inf")``.
        """
        # Compute gain percentile
        q = 1 - self._percentile / 100.0
        gain_perc = torch.quantile(gains, q, dim=0)

        # Filter for positive entries in gain_perc
        ge_zero = gain_perc >= 0
        if torch.any(ge_zero):
            damping_grid_filtered = self._damping_grid[ge_zero]
            gain_perc_filtered = gain_perc[ge_zero]
            max_idx = torch.argmax(gain_perc_filtered)
            return damping_grid_filtered[max_idx]

            # gain_median_filtered = torch.quantile(gains, 0.5, dim=0)[ge_zero]
            # max_idx = torch.argmax(gain_median_filtered)
            # return damping_grid_filtered[max_idx]
        else:
            return float("inf")

    def compute_damping(
        self, first_derivatives: Tensor, second_derivatives: Tensor
    ) -> Tensor:
        """Determine damping parameter for each direction.

        Let ``N₁`` and ``N₂`` denote the number of samples used for computing first-
        and second-order derivatives respectively. Let ``D`` be the number of
        directions.

        Args:
            first_derivatives: 2d tensor of shape ``[N₁,D]`` with the
                gradient projections ``γ[n, d]`` of sample ``n`` along direction ``d``.
            second_derivatives: 2d tensor of shape ``[N₂, D]`` with the
                curvature projections ``λ[n, d]`` of sample ``n`` along direction ``d``.

        Returns:
            1d tensor of shape ``[D]`` with dampings ``δ[d]`` along direction ``d``.
        """
        D = first_derivatives.shape[1]
        num_dampings = len(self._damping_grid)
        device = first_derivatives.device

        self._damping_grid = self._damping_grid.to(device)

        # Vector for dampings for each direction
        dampings = torch.zeros(D, device=device)

        # Make sure that N_1 = N_2
        assert first_derivatives.shape[0] == second_derivatives.shape[0]

        for D_idx in range(D):

            # Extract first and second derivatives for current direction
            first = first_derivatives[:, D_idx]
            second = second_derivatives[:, D_idx]

            # Determine the step for this direction
            step = -torch.mean(first) / (torch.mean(second) + self._damping_grid)

            # Create gain samples for every delta in self._damping_grid
            gains = torch.zeros(self._num_resamples, num_dampings).to(device)

            for resample_idx in range(self._num_resamples):

                # Sample one index vector and evaluate both the gammas and lambdas
                N = first.numel()  # = second.numel()
                rand_idx = torch.randint(low=0, high=N, size=(N,))
                gam_hat_re = torch.mean(first[rand_idx])
                lam_hat_re = torch.mean(second[rand_idx])

                # Compute gain and store sample in gains
                gain = -gam_hat_re * step - 0.5 * lam_hat_re * step**2
                gains[resample_idx, :] = gain

            # Compute damping based on gains
            dampings[D_idx] = self._delta_policy(gains)

        return dampings
