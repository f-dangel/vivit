"""

Note:
    This file is (almost) a copy of
    https://github.com/f-dangel/backpack/blob/development/test/extensions/implementation/base.py#L1-L33 # noqa: B950
"""

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
