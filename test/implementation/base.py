"""

Note:
    This file is (almost) a copy of
    https://github.com/f-dangel/backpack/blob/development/test/extensions/implementation/base.py#L1-L33 # noqa: B950
"""

import numpy
import torch


class ExtensionsImplementation:
    """Base class for autograd and BackPACK implementations of extensions."""

    def __init__(self, problem):
        self.problem = problem

    def gram_sqrt_ggn(self):
        """Generalized Gauss-Newton Gram matrix."""
        raise NotImplementedError

    def sqrt_ggn(self):
        """Square root decomposition of the generalized Gauss-Newton matrix."""
        raise NotImplementedError

    def centered_gram_batch_grad(self):
        """Centered gradient gram matrix."""
        raise NotImplementedError

    def gram_batch_grad(self):
        """Uncentered gradient gram matrix."""
        raise NotImplementedError

    def cov_batch_grad(self):
        """Uncentered gradient covariance matrix."""
        batch_grad_flat = self._batch_grad_flat()
        return torch.einsum("ni,nj->ij", batch_grad_flat, batch_grad_flat)

    def centered_cov_batch_grad(self):
        """Centered gradient covariance matrix."""
        batch_grad_flat = self._batch_grad_flat()
        batch_grad_flat -= batch_grad_flat.mean(0)
        return torch.einsum("ni,nj->ij", batch_grad_flat, batch_grad_flat)

    def centered_batch_grad(self):
        """Centered individual gradients."""
        raise NotImplementedError

    def batch_grad(self):
        """Individual gradients."""
        raise NotImplementedError

    def _batch_grad_flat(self):
        """Compute concatenated flattened individual gradients."""
        batch_grad = self.batch_grad()
        return torch.cat([g.flatten(start_dim=1) for g in batch_grad], dim=1)

    def batch_l2_grad(self):
        """L2 norm of Individual gradients."""
        raise NotImplementedError

    def sgs(self):
        """Sum of Square of Individual gradients"""
        raise NotImplementedError

    def variance(self):
        """Variance of Individual gradients"""
        raise NotImplementedError

    def diag_ggn(self):
        """Diagonal of Gauss Newton"""
        raise NotImplementedError

    def diag_ggn_mc(self, mc_samples):
        """MC approximation of Diagonal of Gauss Newton"""
        raise NotImplementedError

    def diag_h(self):
        """Diagonal of Hessian"""
        raise NotImplementedError

    def ggn(self):
        """Generalized Gauss-Newton matrix."""
        raise NotImplementedError

    def ggn_mc(self, mc_samples):
        """MC approximation of the Generalized Gauss-Newton matrix."""
        raise NotImplementedError

    def _mean_reduction(self):
        """Assert reduction of loss function to be ``'mean'``."""
        N = self.problem.input.shape[0]
        reduction_factor = self.problem.compute_reduction_factor()

        print(1 / N)
        print(reduction_factor)
        assert numpy.isclose(1.0 / N, reduction_factor), "Reduction is not 'mean'"

        return N, reduction_factor

    def _ggn_rank(self, subsampling=None):
        """Return the GGN's rank."""
        D = sum(p.numel() for p in self.problem.model.parameters())

        _, output, _ = self.problem.forward_pass()
        C = output[0].numel()

        if subsampling is None:
            N = self.problem.input.shape[0]
            num_evals = C * N
        else:
            num_evals = C * len(subsampling)

        return min(num_evals, D)

    def _ggn_convert_to_top_k(self, top_space, ggn_subsampling=None):
        """Convert argument specifying the top eigenspace into an absolute number.

        Args:
            top_space (float or int): If integer, describes the absolute number of top
                non-trivial eigenvalues to be considered at most. If float, describes
                the relative number (ratio between 0. and 1., relative to the nontrivial
                eigenspace) of leading eigenvectors that will be used as directions.
                Uses at least one, and at most all nontrivial eigenvalues.
            ggn_subsampling ([int]): Mini-batch indices of samples used in the GGN.
                ``None`` uses the full batch.

        Returns:
            int: Absolute number of top eigenvalues to be considered.

        Raises:
            ValueError: If the input is of incorrect type
        """
        nontrivial_evals = self._ggn_rank(subsampling=ggn_subsampling)

        if isinstance(top_space, int):
            k = top_space
        elif isinstance(top_space, float):
            k = int(top_space * nontrivial_evals)
        else:
            raise ValueError(f"Input must be int or float. Got {top_space}")

        # clip to below by 1 and above by number of nontrivial eigenvalues
        k = min(nontrivial_evals, max(k, 1))

        return k
