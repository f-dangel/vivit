from test.implementation.base import ExtensionsImplementation
from typing import List, Tuple

import backpack.extensions as new_ext
import torch
from backpack import backpack
from backpack.extensions import SqrtGGNExact, SqrtGGNMC
from torch import Tensor, zeros_like
from torch.linalg import eigh

from vivit.extensions.firstorder.batch_grad.gram_batch_grad import (
    CenteredBatchGrad,
    CenteredGramBatchGrad,
    GramBatchGrad,
)
from vivit.extensions.secondorder.sqrt_ggn.gram_sqrt_ggn import (
    GramSqrtGGNExact,
    GramSqrtGGNMC,
)
from vivit.extensions.secondorder.vivit import ViViTGGNExact, ViViTGGNMC
from vivit.utils.gram import reshape_as_square


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

    def vivit_ggn_mc_mat_prod_chunk(
        self,
        mat: List[Tensor],
        mc_samples: int,
        chunks: int = 10,
        subsampling: List[int] = None,
    ) -> List[Tensor]:
        """Like ``vivit_ggn_mat_prod``, but handles larger sample number by chunking.

        Uses ``ViViTGGNMC`` extension.

        Args:
            mat: Stacked vectors in parameter format.
            mc_samples: Number of Monte-Carlo samples to approximate the loss Hessian.
            chunks: Number of sequential computations the work will be split up.
            subsampling: Indices of samples to use for the computation.
                Default: ``None``.

        """
        chunk_samples = self.chunk_sizes(mc_samples, chunks)
        chunk_weights = [samples / mc_samples for samples in chunk_samples]

        ggn_mat_prod = [zeros_like(m) for m in mat]

        for weight, samples in zip(chunk_weights, chunk_samples):
            for idx, ggn_mat in enumerate(
                self.vivit_ggn_mat_prod(
                    mat, subsampling=subsampling, mc_samples=samples
                )
            ):
                ggn_mat_prod[idx] += weight * ggn_mat

        return ggn_mat_prod

    def vivit_ggn_mat_prod(
        self, mat: List[Tensor], subsampling: List[int] = None, mc_samples: int = 0
    ) -> List[Tensor]:
        """Multiply each vector in ``mat`` by the GGN.

        Uses ``ViViTGGNExact`` extension if ``mc_samples =0``, otherwise ``ViViTGGNMC``.

        Args:
            mat: Stacked vectors in parameter format.
            subsampling: Indices of samples to use for the computation.
                Default: ``None``.
            mc_samples: Number of Monte-Carlo samples to approximate the loss Hessian.
                ``0`` indicates using the exact representation.

        Returns:
            Stacked results of GGN-vector products in parameter format.
        """
        if mc_samples == 0:
            extension = ViViTGGNExact(subsampling=subsampling)
        else:
            extension = ViViTGGNMC(subsampling=subsampling, mc_samples=mc_samples)
        savefield = extension.savefield

        with backpack(extension):
            _, _, loss = self.problem.forward_pass()
            loss.backward()

        V_t_mat = sum(
            getattr(p, savefield)["V_t_mat_prod"](m)
            for p, m in zip(self.problem.model.parameters(), mat)
        )
        V_V_t_mat = [
            getattr(p, savefield)["V_mat_prod"](V_t_mat)
            for p in self.problem.model.parameters()
        ]

        # adapt scaling in case of sub-sampling
        V_V_t_mat = [
            self.subsampling_correction(subsampling=subsampling) * mat
            for mat in V_V_t_mat
        ]

        return V_V_t_mat

    def vivit_ggn_eigh(
        self, subsampling: List[int] = None
    ) -> Tuple[Tensor, List[Tensor]]:
        """Compute GGN eigenvalues and eigenvectors from the Gram matrix.

        Uses ``ViViTGGNExact`` extension.

        Args:
            subsampling: Indices of samples to use for the computation.
                Default: ``None``.

        Returns:
            Eigenvalues and stacked unnormalized eigenvectors in parameter format.
        """
        extension = ViViTGGNExact(subsampling=subsampling)
        savefield = extension.savefield

        with backpack(extension):
            _, _, loss = self.problem.forward_pass()
            loss.backward()

        gram = sum(
            getattr(p, savefield)["gram_mat"]() for p in self.problem.model.parameters()
        )
        gram *= self.subsampling_correction(subsampling=subsampling)
        gram_as_square = reshape_as_square(gram)

        gram_evals, gram_evecs = eigh(gram_as_square)
        gram_evecs = gram_evecs.transpose(0, 1).reshape(-1, *gram.shape[:2])

        evecs = [
            getattr(p, savefield)["V_mat_prod"](gram_evecs)
            for p in self.problem.model.parameters()
        ]

        return gram_evals, evecs

    def subsampling_correction(self, subsampling: List[int] = None) -> float:
        """Determine the correction factor for reduction if sub-sampling is enabled.

        Args:
            subsampling: Indices of samples used for the computation.
                Default: ``None``.

        Returns:
            Correction factor to fix the scaling with sub-sampling.
        """
        if subsampling is None:
            factor = 1.0
        else:
            reduction_str = self.problem.reduction_string()
            batch_size = self.problem.input.shape[0]
            factor = {
                "mean": batch_size / len(subsampling),
                "sum": 1.0,
            }[reduction_str]

        return factor
