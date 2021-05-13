import warnings

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

    @staticmethod
    def _degeneracy_warning(evals):
        """Warn if eigenvalues are degenerate and directions not unique.

        ``evals`` are assumed to be sorted.
        """
        degeneracy = torch.isclose(evals[1:], evals[:-1])
        if torch.any(degeneracy):
            warnings.warn(
                "Eigenvalue degeneracy detected!"
                + " This usually leads to failing tests as directions are not unique."
                + f"\nGot eigenvalues:\n{evals}"
                + f"\nDegeneracy with neighboring eigenvalue detected:\n{degeneracy}"
            )


def parameter_groups_to_idx(param_groups, parameters):
    """Return indices for parameter groups in parameters."""
    params_in_group_ids = [id(p) for group in param_groups for p in group["params"]]
    params_ids = [id(p) for p in parameters]

    if len(params_in_group_ids) != len(set(params_in_group_ids)):
        raise ValueError("Same parameters occur in different groups.")
    if sorted(params_in_group_ids) != sorted(params_ids):
        raise ValueError("Parameters and group parameters don't match.")

    num_params = [param.numel() for param in parameters]
    param_ids = [id(p) for p in parameters]
    indices = []

    for group in param_groups:
        param_indices = []

        for param in group["params"]:
            param_idx = param_ids.index(id(param))

            start = sum(num_params[:param_idx])
            end = sum(num_params[: param_idx + 1])
            param_indices += list(range(start, end))

        indices.append(param_indices)

    return indices


def parameter_groups_to_param_idx(param_groups, parameters):
    """Return indices for parameters in parameter groups w.r.t parameters."""
    params_in_group_ids = [id(p) for group in param_groups for p in group["params"]]
    params_ids = [id(p) for p in parameters]

    if len(params_in_group_ids) != len(set(params_in_group_ids)):
        raise ValueError("Same parameters occur in different groups.")
    if sorted(params_in_group_ids) != sorted(params_ids):
        raise ValueError("Parameters and group parameters don't match.")

    param_ids = [id(p) for p in parameters]
    indices = []

    for group in param_groups:
        param_indices = []

        for param in group["params"]:
            param_indices.append(param_ids.index(id(param)))

        indices.append(param_indices)

    return indices
