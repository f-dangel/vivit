"""BackPACK implementation of operations used in ``vivit.optim``."""

from test.implementation.backpack import BackpackExtensions

from backpack import backpack

from vivit.optim import DirectionalDerivativesComputation


class BackpackOptimExtensions(BackpackExtensions):
    def gammas_ggn(
        self, param_groups, subsampling_directions=None, subsampling_first=None
    ):
        """First-order directional derivatives along leading GGN eigenvectors via
        ``vivit.optim.computations``.

        Args:
            param_groups ([dict]): Parameter groups like for ``torch.nn.Optimizer``s.
            subsampling_directions ([int] or None): Indices of samples used to compute
                Newton directions. If ``None``, all samples in the batch will be used.
            subsampling_first ([int], optional): Sample indices used for individual
                gradients.
        """
        computations = DirectionalDerivativesComputation(
            subsampling_directions=subsampling_directions,
            subsampling_first=subsampling_first,
        )

        _, _, loss = self.problem.forward_pass()

        with backpack(
            *computations.get_extensions(param_groups),
            extension_hook=computations.get_extension_hook(
                param_groups,
                keep_backpack_buffers=False,
                keep_gram_mat=False,
                keep_gram_evecs=False,
                keep_batch_size=False,
                keep_gram_evals=False,
            ),
        ):
            loss.backward()

        return [computations._gammas[id(group)] for group in param_groups]

    def lambdas_ggn(
        self, param_groups, subsampling_directions=None, subsampling_second=None
    ):
        """Second-order directional derivatives along leading GGN eigenvectors via
        ``vivit.optim.computations``.

        Args:
            param_groups ([dict]): Parameter groups like for ``torch.nn.Optimizer``s.
            subsampling_directions ([int] or None): Indices of samples used to compute
                Newton directions. If ``None``, all samples in the batch will be used.
            subsampling_second ([int], optional): Sample indices used for individual
                curvature matrices.
        """
        computations = DirectionalDerivativesComputation(
            subsampling_directions=subsampling_directions,
            subsampling_second=subsampling_second,
        )

        _, _, loss = self.problem.forward_pass()

        with backpack(
            *computations.get_extensions(param_groups),
            extension_hook=computations.get_extension_hook(
                param_groups,
                keep_backpack_buffers=False,
                keep_gram_mat=False,
                keep_gram_evecs=False,
                keep_batch_size=False,
                keep_gram_evals=False,
            ),
        ):
            loss.backward()

        return [computations._lambdas[id(group)] for group in param_groups]
