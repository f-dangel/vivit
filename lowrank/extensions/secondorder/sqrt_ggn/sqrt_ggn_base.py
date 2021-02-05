import torch

from backpack.extensions.mat_to_mat_jac_base import MatToJacMat


class SqrtGGNBaseModule(MatToJacMat):
    def __init__(self, derivatives, params=None):
        super().__init__(derivatives, params=params)

    def bias(self, ext, module, g_inp, g_out, backproped):
        return self.derivatives.bias_jac_t_mat_prod(
            module, g_inp, g_out, backproped, sum_batch=False
        )

    def weight(self, ext, module, g_inp, g_out, backproped):
        return self.derivatives.weight_jac_t_mat_prod(
            module, g_inp, g_out, backproped, sum_batch=False
        )
