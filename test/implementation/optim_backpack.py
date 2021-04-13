"""BackPACK implementation of operations used in ``lowrank.optim``."""

from test.implementation.backpack import BackpackExtensions

from backpack import backpack

from lowrank.optim.computations import BaseComputations
from lowrank.optim.damped_newton import DampedNewton


class BackpackOptimExtensions(BackpackExtensions):
    def gammas_ggn(self, top_k, subsampling_directions=None, subsampling_first=None):
        """First-order directional derivatives along leading GGN eigenvectors via
        ``lowrank.optim.computations``.

        Args:
            top_k (int): Number of leading eigenvectors used as directions. Will be
                clipped to ``[1, max]`` with ``max`` the maximum number of nontrivial
                eigenvalues.
            subsampling_directions ([int] or None): Indices of samples used to compute
                Newton directions. If ``None``, all samples in the batch will be used.
            subsampling_first ([int], optional): Sample indices used for individual
                gradients.
        """
        k = self._ggn_convert_to_top_k(top_k)

        param_groups = self._param_groups_top_k_criterion(k)
        computations = BaseComputations(
            subsampling_directions=subsampling_directions,
            subsampling_first=subsampling_first,
        )

        _, _, loss = self.problem.forward_pass()

        with backpack(
            *computations.get_extensions(param_groups),
            extension_hook=computations.get_extension_hook(param_groups),
        ):
            loss.backward()

        for group in param_groups:
            computations._eval_directions(group)
            computations._filter_directions(group)
            computations._eval_gammas(group)

        return list(computations._gammas.values())[0]

    def lambdas_ggn(self, top_k, subsampling_directions=None, subsampling_second=None):
        """Second-order directional derivatives along leading GGN eigenvectors via
        ``lowrank.optim.computations``.

        Args:
            top_k (int): Number of leading eigenvectors used as directions. Will be
                clipped to ``[1, max]`` with ``max`` the maximum number of nontrivial
                eigenvalues.
            subsampling_directions ([int] or None): Indices of samples used to compute
                Newton directions. If ``None``, all samples in the batch will be used.
            subsampling_second ([int], optional): Sample indices used for individual
                curvature matrices.
        """
        k = self._ggn_convert_to_top_k(top_k)

        param_groups = self._param_groups_top_k_criterion(k)
        computations = BaseComputations(
            subsampling_directions=subsampling_directions,
            subsampling_second=subsampling_second,
        )

        _, _, loss = self.problem.forward_pass()

        with backpack(
            *computations.get_extensions(param_groups),
            extension_hook=computations.get_extension_hook(param_groups),
        ):
            loss.backward()

        for group in param_groups:
            computations._eval_directions(group)
            computations._filter_directions(group)
            computations._eval_lambdas(group)

        return list(computations._lambdas.values())[0]

    def newton_step(
        self,
        top_k,
        damping,
        subsampling_directions=None,
        subsampling_first=None,
        subsampling_second=None,
    ):
        """Directionally-damped Newton step along the top-k GGN eigenvectors.

        Args:
            top_k (int): Number of leading eigenvectors used as directions. Will be
                clipped to ``[1, max]`` with ``max`` the maximum number of nontrivial
                eigenvalues.
            damping (lowrank.optim.damping.BaseDamping): Policy for selecting
                dampings along a direction from first- and second- order directional
                derivatives.
            subsampling_directions ([int] or None): Indices of samples used to compute
                Newton directions. If ``None``, all samples in the batch will be used.
            subsampling_first ([int], optional): Sample indices used for individual
                gradients.
            subsampling_second ([int], optional): Sample indices used for individual
                curvature matrices.
        """
        k = self._ggn_convert_to_top_k(top_k)

        param_groups = self._param_groups_top_k_criterion(k)
        computations = BaseComputations(
            subsampling_directions=subsampling_directions,
            subsampling_first=subsampling_first,
            subsampling_second=subsampling_second,
        )

        _, _, loss = self.problem.forward_pass()

        with backpack(
            *computations.get_extensions(param_groups),
            extension_hook=computations.get_extension_hook(param_groups),
        ):
            loss.backward()

        for group in param_groups:
            computations._eval_directions(group)
            computations._filter_directions(group)
            computations._eval_gammas(group)
            computations._eval_lambdas(group)
            computations._eval_deltas(group, damping)
            computations._eval_newton_step(group)

        return list(computations._newton_step.values())[0]

    def _param_groups_top_k_criterion(self, k):
        """Put all parameters in a single group. Use top-k criterion for directions."""
        top_k = DampedNewton.make_default_criterion(k=k)

        return [{"params": list(self.problem.model.parameters()), "criterion": top_k}]
