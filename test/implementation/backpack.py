"""

Note:
    This file is (almost) a copy of
    https://github.com/f-dangel/backpack/blob/development/test/extensions/implementation/backpack.py#L1-L84 # noqa: B950
"""
import math
from test.implementation.base import ExtensionsImplementation

import backpack.extensions as new_ext
import torch
from backpack import backpack

from lowrank.extensions.firstorder.batch_grad.gram_batch_grad import (
    CenteredBatchGrad,
    CenteredGramBatchGrad,
    GramBatchGrad,
)
from lowrank.extensions.secondorder.sqrt_ggn import SqrtGGNExact, SqrtGGNMC
from lowrank.extensions.secondorder.sqrt_ggn.gram_sqrt_ggn import (
    GramSqrtGGNExact,
    GramSqrtGGNMC,
)
from lowrank.utils.ggn import V1_t_V2, V_mat_prod, V_t_mat_prod, V_t_V
from lowrank.utils.gram import reshape_as_square


class BackpackExtensions(ExtensionsImplementation):
    """Extension implementations with BackPACK."""

    def __init__(self, problem):
        problem.extend()
        super().__init__(problem)

    def centered_batch_grad(self):
        hook = CenteredBatchGrad()

        with backpack(new_ext.BatchGrad(), extension_hook=hook):
            _, _, loss = self.problem.forward_pass()
            loss.backward()

        return [p.centered_grad_batch for p in self.problem.model.parameters()]

    def gram_sqrt_ggn_mc(self, mc_samples, layerwise=False, free_sqrt_ggn=False):
        hook = GramSqrtGGNMC(layerwise=layerwise, free_sqrt_ggn=free_sqrt_ggn)

        with backpack(SqrtGGNMC(mc_samples=mc_samples), extension_hook=hook):
            _, _, loss = self.problem.forward_pass()
            loss.backward()

        return hook.get_result()

    def gram_sqrt_ggn(self, layerwise=False, free_sqrt_ggn=False):
        hook = GramSqrtGGNExact(layerwise=layerwise, free_sqrt_ggn=free_sqrt_ggn)

        with backpack(SqrtGGNExact(), extension_hook=hook):
            _, _, loss = self.problem.forward_pass()
            loss.backward()

        return hook.get_result()

    def ggn_mc(self, mc_samples, subsampling=None):
        sqrt_ggn_mc = self.sqrt_ggn_mc(mc_samples, subsampling=subsampling)
        return self._square_sqrt_ggn(sqrt_ggn_mc)

    def ggn(self, subsampling=None):
        sqrt_ggn = self.sqrt_ggn(subsampling=subsampling)
        return self._square_sqrt_ggn(sqrt_ggn)

    def ggn_mc_chunk(self, mc_samples, chunks=10, subsampling=None):
        """Like ``ggn_mc``, but handles larger number of samples by chunking."""
        chunk_samples = self.chunk_sizes(mc_samples, chunks)
        chunk_weights = [samples / mc_samples for samples in chunk_samples]

        ggn_mc = None

        for weight, samples in zip(chunk_weights, chunk_samples):
            chunk_ggn_mc = weight * self.ggn_mc(samples, subsampling=subsampling)
            ggn_mc = chunk_ggn_mc if ggn_mc is None else ggn_mc + chunk_ggn_mc

        return ggn_mc

    @staticmethod
    def _square_sqrt_ggn(sqrt_ggn):
        """Utility function to concatenate and square the GGN factorization."""
        sqrt_ggn = torch.cat([s.flatten(start_dim=2) for s in sqrt_ggn], dim=2)
        return torch.einsum("nci,ncj->ij", sqrt_ggn, sqrt_ggn)

    def sqrt_ggn(self, subsampling=None):
        with backpack(SqrtGGNExact(subsampling=subsampling)):
            _, _, loss = self.problem.forward_pass()
            loss.backward()

        return [p.sqrt_ggn_exact for p in self.problem.model.parameters()]

    def sqrt_ggn_mc(self, mc_samples, subsampling=None):
        with backpack(SqrtGGNMC(mc_samples=mc_samples, subsampling=subsampling)):
            _, _, loss = self.problem.forward_pass()
            loss.backward()

        return [p.sqrt_ggn_mc for p in self.problem.model.parameters()]

    def centered_gram_batch_grad(self, layerwise=False, free_grad_batch=False):
        hook = CenteredGramBatchGrad(
            layerwise=layerwise, free_grad_batch=free_grad_batch
        )

        with backpack(new_ext.BatchGrad(), extension_hook=hook):
            _, _, loss = self.problem.forward_pass()
            loss.backward()

        return hook.get_result()

    def gram_batch_grad(self, layerwise=False, free_grad_batch=False):
        hook = GramBatchGrad(layerwise=layerwise, free_grad_batch=free_grad_batch)

        with backpack(new_ext.BatchGrad(), extension_hook=hook):
            _, _, loss = self.problem.forward_pass()
            loss.backward()

        return hook.get_result()

    def batch_grad(self, subsampling=None):
        with backpack(new_ext.BatchGrad(subsampling=subsampling)):
            _, _, loss = self.problem.forward_pass()
            loss.backward()
            batch_grads = [p.grad_batch for p in self.problem.model.parameters()]
        return batch_grads

    def batch_l2_grad(self):
        with backpack(new_ext.BatchL2Grad()):
            _, _, loss = self.problem.forward_pass()
            loss.backward()
            batch_l2_grad = [p.batch_l2 for p in self.problem.model.parameters()]
        return batch_l2_grad

    def sgs(self):
        with backpack(new_ext.SumGradSquared()):
            _, _, loss = self.problem.forward_pass()
            loss.backward()
            sgs = [p.sum_grad_squared for p in self.problem.model.parameters()]
        return sgs

    def variance(self):
        with backpack(new_ext.Variance()):
            _, _, loss = self.problem.forward_pass()
            loss.backward()
            variances = [p.variance for p in self.problem.model.parameters()]
        return variances

    def diag_ggn(self):
        with backpack(new_ext.DiagGGNExact()):
            _, _, loss = self.problem.forward_pass()
            loss.backward()
            diag_ggn = [p.diag_ggn_exact for p in self.problem.model.parameters()]
        return diag_ggn

    def diag_ggn_mc(self, mc_samples):
        with backpack(new_ext.DiagGGNMC(mc_samples=mc_samples)):
            _, _, loss = self.problem.forward_pass()
            loss.backward()
            diag_ggn_mc = [p.diag_ggn_mc for p in self.problem.model.parameters()]
        return diag_ggn_mc

    def diag_h(self):
        with backpack(new_ext.DiagHessian()):
            _, _, loss = self.problem.forward_pass()
            loss.backward()
            diag_h = [p.diag_h for p in self.problem.model.parameters()]
        return diag_h

    def ggn_mat_prod(self, mat_list, subsampling=None):
        """Vectorized multiplication with the Generalized Gauss-Newton/Fisher.

        Uses multiplication with symmetric factors ``V``, ``Vᵀ``, and ``G = V @ Vᵀ``.

        Args:
            mat_list ([torch.Tensor]): Layer-wise split of matrices to be multiplied
                by the GGN. Each item has a free leading dimension, and shares the
                same trailing dimensions with the associated parameter.
            subsampling ([int]): Indices of samples in the mini-batch for which
                the GGN/Fisher should be multiplied with. ``None`` uses the
                entire mini-batch.

        Returns:
            [torch.Tensor]: Result of multiplication with the GGN
        """
        with backpack(SqrtGGNExact()):
            _, _, loss = self.problem.forward_pass()
            loss.backward()

        return self._V_V_t_mat_prod(mat_list, "sqrt_ggn_exact", subsampling=subsampling)

    def _V_V_t_mat_prod(self, mat_list, savefield, subsampling=None):
        """Multiply with the GGN's symmetric factors ``V`` and ``Vᵀ``.

        Args:
            mat_list ([torch.Tensor]): Layer-wise split of matrices to be multiplied
                by the GGN. Each item has a free leading dimension, and shares the
                same trailing dimensions with the associated parameter.
            savefield (str): Attribute under which ``Vᵀ`` is saved in the parameters.
            subsampling ([int]): Indices of samples in the mini-batch for which
                the GGN/Fisher should be multiplied with. ``None`` uses the
                entire mini-batch.

        Returns:
            [torch.Tensor]: Result of multiplication with ``V @ Vᵀ``.
        """
        parameters = list(self.problem.model.parameters())

        result = V_t_mat_prod(mat_list, parameters, savefield, subsampling=subsampling)
        result = V_mat_prod(result, parameters, savefield, subsampling=subsampling)

        return result

    def ggn_mc_mat_prod(self, mat_list, mc_samples, subsampling=None):
        """Vectorized multiplication with the MC Generalized Gauss-Newton/Fisher.

        Uses multiplication with symmetric factors ``V``, ``Vᵀ``, and ``G = V @ Vᵀ``.

        Args:
            mat_list ([torch.Tensor]): Layer-wise split of matrices to be multiplied
                by the GGN. Each item has a free leading dimension, and shares the
                same trailing dimensions with the associated parameter.
            mc_samples (int): Number of MC samples used to approximate the GGN.
            subsampling ([int]): Indices of samples in the mini-batch for which
                the GGN/Fisher should be multiplied with. ``None`` uses the
                entire mini-batch.

        Returns:
            [torch.Tensor]: Result of multiplication with the MC-approximated GGN
        """
        with backpack(SqrtGGNMC(mc_samples=mc_samples)):
            _, _, loss = self.problem.forward_pass()
            loss.backward()

        return self._V_V_t_mat_prod(mat_list, "sqrt_ggn_mc", subsampling=subsampling)

    def ggn_mc_mat_prod_chunk(self, mat_list, mc_samples, chunks=10, subsampling=None):
        """Like ``ggn_mc_mat_prod``. Handles larger number of samples by chunking."""
        chunk_samples = self.chunk_sizes(mc_samples, chunks)
        chunk_weights = [samples / mc_samples for samples in chunk_samples]

        ggn_mc_mat = [None for _ in mat_list]

        for weight, samples in zip(chunk_weights, chunk_samples):
            chunk_ggn_mc_mat = self.ggn_mc_mat_prod(
                mat_list, samples, subsampling=subsampling
            )
            chunk_ggn_mc_mat = [weight * ggn_mc_m for ggn_mc_m in chunk_ggn_mc_mat]

            # update existing ggn_mc_mats
            for idx, res in enumerate(chunk_ggn_mc_mat):
                old_res = ggn_mc_mat[idx]
                new_res = res if old_res is None else old_res + res

                ggn_mc_mat[idx] = new_res

        return ggn_mc_mat

    @staticmethod
    def chunk_sizes(total_size, num_chunks):
        """Return list containing the sizes of chunks."""
        chunk_size = max(total_size // num_chunks, 1)

        if chunk_size == 1:
            sizes = total_size * [chunk_size]
        else:
            equal, rest = divmod(total_size, chunk_size)
            sizes = equal * [chunk_size]

            if rest != 0:
                sizes.append(rest)

        return sizes

    def gammas_ggn(self, top_space, ggn_subsampling=None, grad_subsampling=None):
        """First-order derivatives ``γ[n, d]`` along the leading GGN eigenvectors.

        Args:
            top_space (float or int): If integer, describes the absolute number of top
                non-trivial eigenvalues to be considered at most. If float, describes
                the relative number (ratio between 0. and 1., relative to the nontrivial
                eigenspace) of leading eigenvectors that will be used as directions.
                Uses at least one, and at most all nontrivial eigenvalues.
            ggn_subsampling ([int], optional): Sample indices used for the GGN.
            grad_subsampling ([int], optional): Sample indices used for individual
                gradients.

        Returns:
            torch.Tensor: 2d tensor containing ``γ[n, d]``.
        """
        N, _ = self._mean_reduction()

        # create savefield buffers
        self.sqrt_ggn()
        savefield = "sqrt_ggn_exact"
        gram = V_t_V(
            self.problem.model.parameters(), savefield, subsampling=ggn_subsampling
        )

        # compensate subsampling scale
        if ggn_subsampling is not None:
            gram *= N / len(ggn_subsampling)

        evals, evecs = reshape_as_square(gram).symeig(eigenvectors=True)

        # select top eigenspace
        k = self._ggn_convert_to_top_k(top_space, ggn_subsampling=ggn_subsampling)
        evals = evals[-k:]
        evecs = evecs[:, -k:]

        # flattened individual gradients
        grad_batch = self.batch_grad(subsampling=grad_subsampling)

        # compensate individual gradient scaling from BackPACK
        individual_gradients = [g * N for g in grad_batch]

        V_t_g = V_t_mat_prod(
            individual_gradients,
            list(self.problem.model.parameters()),
            savefield,
            subsampling=ggn_subsampling,
            flatten=True,
        )

        # compensate subsampling scale from multiplication with ``Vᵀ``
        if ggn_subsampling is not None:
            V_t_g *= math.sqrt(N / len(ggn_subsampling))

        gammas = torch.einsum("ni,id->nd", V_t_g, evecs) / evals.sqrt()

        return gammas

    def lambdas_ggn(self, top_space, ggn_subsampling=None, lambda_subsampling=None):
        """Second-order derivatives ``λ[n, d]`` along the leading GGN eigenvectors.

        Uses the exact GGN for λ.

        Args:
            top_space (float or int): If integer, describes the absolute number of top
                non-trivial eigenvalues to be considered at most. If float, describes
                the relative number (ratio between 0. and 1., relative to the nontrivial
                eigenspace) of leading eigenvectors that will be used as directions.
                Uses at least one, and at most all nontrivial eigenvalues.
            ggn_subsampling ([int], optional): Sample indices used for the GGN.
            lambda_subsampling ([int], optional): Sample indices used for lambdas.

        Returns:
            torch.Tensor: 2d tensor containing ``λ[n, d]``.
        """
        N, _ = self._mean_reduction()

        # create savefield buffers
        self.sqrt_ggn()
        savefield = "sqrt_ggn_exact"
        gram = V_t_V(
            self.problem.model.parameters(), savefield, subsampling=ggn_subsampling
        )

        # compensate subsampling scale
        if ggn_subsampling is not None:
            gram *= N / len(ggn_subsampling)

        evals, evecs = reshape_as_square(gram).symeig(eigenvectors=True)

        # select top eigenspace
        k = self._ggn_convert_to_top_k(top_space, ggn_subsampling=ggn_subsampling)
        evals = evals[-k:]
        evecs = evecs[:, -k:]

        V_n_t_V = V1_t_V2(
            self.problem.model.parameters(),
            savefield,
            savefield,
            subsampling1=lambda_subsampling,
            subsampling2=ggn_subsampling,
        )
        V_n_t_V = V_n_t_V.flatten(start_dim=2)

        # compensate scale
        scale = math.sqrt(N)

        if ggn_subsampling is not None:
            scale *= math.sqrt(N / len(ggn_subsampling))

        V_n_t_V *= scale

        # compute lambdas
        V_n_t_V_evecs = torch.einsum("cni,id->cnd", V_n_t_V, evecs)
        C_axis = 0

        lambdas = (V_n_t_V_evecs ** 2).sum(C_axis) / evals

        return lambdas
