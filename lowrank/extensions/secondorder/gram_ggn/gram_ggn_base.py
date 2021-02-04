import torch

from backpack.extensions.mat_to_mat_jac_base import MatToJacMat


class GramGGNBaseModule(MatToJacMat):
    def __init__(self, derivatives, params=None):
        super().__init__(derivatives, params=params)

    def bias(self, ext, module, g_inp, g_out, backproped):
        sqrt_gram = self.derivatives.bias_jac_t_mat_prod(
            module, g_inp, g_out, backproped, sum_batch=False
        )
        return self.pairwise_dot(sqrt_gram)

    def weight(self, ext, module, g_inp, g_out, backproped):
        sqrt_gram = self.derivatives.weight_jac_t_mat_prod(
            module, g_inp, g_out, backproped, sum_batch=False
        )
        return self.pairwise_dot(sqrt_gram)

    @staticmethod
    def pairwise_dot(tensor):
        """Compute pairwise scalar product. Pairs are the two leading dims."""
        out_dim = 2 * (tensor.shape[0] * tensor.shape[1],)

        # TODO Avoid flattening with more sophisticated einsum equation
        tensor_flat = tensor.flatten(start_dim=2)
        equation = "ijf,klf->ijkl"

        return torch.einsum(equation, tensor_flat, tensor_flat).reshape(out_dim)
