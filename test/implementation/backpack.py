"""

Note:
    This file is (almost) a copy of
    https://github.com/f-dangel/backpack/blob/development/test/extensions/implementation/backpack.py#L1-L84 # noqa: B950
"""
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
        chunk_samples = (chunks - 1) * [mc_samples // chunks]
        last_samples = mc_samples - sum(chunk_samples)
        if last_samples != 0:
            chunk_samples.append(last_samples)

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

    def batch_grad(self):
        with backpack(new_ext.BatchGrad()):
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
