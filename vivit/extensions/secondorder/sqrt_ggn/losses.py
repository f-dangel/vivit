from functools import partial

from backpack.core.derivatives.crossentropyloss import CrossEntropyLossDerivatives
from backpack.core.derivatives.mseloss import MSELossDerivatives
from backpack.extensions.secondorder.hbp import LossHessianStrategy

from vivit.extensions.secondorder.sqrt_ggn.sqrt_ggn_base import SqrtGGNBaseModule


class SqrtGGNLoss(SqrtGGNBaseModule):
    def backpropagate(self, ext, module, grad_inp, grad_out, backproped):
        hess_func = self.make_loss_hessian_func(ext)

        return hess_func(module, grad_inp, grad_out)

    def make_loss_hessian_func(self, ext):
        """Get function that produces the backpropagated quantity.

        Args:
            ext (backpack.backprop_extension.BackpropExtension): BackPACK extension
                that is active in the current backward pass.

        Returns:
            callable: Function without arguments, that produces the loss Hessian's
                symmetric factorization.

        Raises:
            ValueError: If the strategy to represent the loss Hessian is unknown.
        """
        loss_hessian_strategy = ext.loss_hessian_strategy
        subsampling = ext.get_subsampling()

        if loss_hessian_strategy == LossHessianStrategy.EXACT:
            return partial(self.derivatives.sqrt_hessian, subsampling=subsampling)
        elif loss_hessian_strategy == LossHessianStrategy.SAMPLING:
            mc_samples = ext.get_num_mc_samples()
            return partial(
                self.derivatives.sqrt_hessian_sampled,
                mc_samples=mc_samples,
                subsampling=subsampling,
            )

        else:
            raise ValueError(
                "Unknown hessian strategy {}".format(loss_hessian_strategy)
            )


class SqrtGGNMSELoss(SqrtGGNLoss):
    def __init__(self):
        super().__init__(derivatives=MSELossDerivatives())


class SqrtGGNCrossEntropyLoss(SqrtGGNLoss):
    def __init__(self):
        super().__init__(derivatives=CrossEntropyLossDerivatives())
