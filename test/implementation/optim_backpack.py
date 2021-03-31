"""BackPACK implementation of operations used in ``lowrank.optim``."""

from test.implementation.backpack import BackpackExtensions

from backpack import backpack

from lowrank.optim.computations import BaseComputations
from lowrank.optim.damped_newton import DampedNewton


class BackpackOptimExtensions(BackpackExtensions):
    def gammas_ggn(self, top_k):
        """First-order directional derivatives along leading GGN eigenvectors via
        ``lowrank.optim.computations``.

        Args:
            top_k (int): Number of leading eigenvectors used as directions. Will be
                clipped to ``[1, max]`` with ``max`` the maximum number of nontrivial
                eigenvalues.
        """
        k = self._ggn_convert_to_top_k(top_k)

        param_groups = self._param_groups_top_k_criterion(k)
        computations = BaseComputations()

        _, _, loss = self.problem.forward_pass()

        with backpack(*computations.get_extensions(param_groups)):
            loss.backward()

        for group in param_groups:
            computations._eval_directions(group)
            computations._filter_directions(group)
            computations._eval_gammas(group)

        return list(computations._gammas.values())[0]

    def lambdas_ggn(self, top_k):
        """Second-order directional derivatives along leading GGN eigenvectors via
        ``lowrank.optim.computations``.

        Args:
            top_k (int): Number of leading eigenvectors used as directions. Will be
                clipped to ``[1, max]`` with ``max`` the maximum number of nontrivial
                eigenvalues.
        """
        k = self._ggn_convert_to_top_k(top_k)

        param_groups = self._param_groups_top_k_criterion(k)
        computations = BaseComputations()

        _, _, loss = self.problem.forward_pass()

        with backpack(*computations.get_extensions(param_groups)):
            loss.backward()

        _, _, loss = self.problem.forward_pass()

        for group in param_groups:
            computations._eval_directions(group)
            computations._filter_directions(group)
            computations._eval_lambdas(group)

        return list(computations._lambdas.values())[0]

    def _param_groups_top_k_criterion(self, k):
        """Put all parameters in a single group. Use top-k criterion for directions."""
        top_k = DampedNewton.make_default_criterion(k=k)

        return [{"params": list(self.problem.model.parameters()), "criterion": top_k}]
