from test.implementation.base import ExtensionsImplementation

import backpack.extensions as new_ext
import torch
from backpack import backpack
from backpack.extensions import SqrtGGNExact, SqrtGGNMC

from vivit.extensions.firstorder.batch_grad.gram_batch_grad import (
    CenteredBatchGrad,
    CenteredGramBatchGrad,
    GramBatchGrad,
)
from vivit.extensions.secondorder.sqrt_ggn.gram_sqrt_ggn import (
    GramSqrtGGNExact,
    GramSqrtGGNMC,
)


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
