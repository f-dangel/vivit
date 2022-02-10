from test.implementation.base import ExtensionsImplementation, parameter_groups_to_idx
from typing import List, Tuple

import torch
from backpack.hessianfree.ggnvp import ggn_vector_product, ggn_vector_product_from_plist
from backpack.utils.convert_parameters import vector_to_parameter_list
from torch import Tensor, zeros_like
from torch.nn import Parameter
from torch.nn.utils.convert_parameters import parameters_to_vector


class AutogradExtensions(ExtensionsImplementation):
    """Extension implementations with autograd."""

    def centered_batch_grad(self):
        N_axis = 0
        centered_batch_grad = [bg - bg.mean(N_axis) for bg in self.batch_grad()]

        return centered_batch_grad

    def centered_gram_batch_grad(self):
        batch_grad_flat = self._batch_grad_flat()
        batch_grad_flat -= batch_grad_flat.mean(0)
        return torch.einsum("if,jf->ij", batch_grad_flat, batch_grad_flat)

    def gram_batch_grad(self):
        batch_grad_flat = self._batch_grad_flat()
        return torch.einsum("if,jf->ij", batch_grad_flat, batch_grad_flat)

    def batch_grad(self, subsampling=None):
        batch_size = self.problem.input.shape[0]

        if subsampling is None:
            subsampling = list(range(batch_size))

        batch_grad = [
            torch.zeros(len(subsampling), *p.size()).to(self.problem.device)
            for p in self.problem.model.parameters()
        ]
        factor = self.problem.compute_reduction_factor()

        for out_idx, n in enumerate(subsampling):
            _, _, loss_n = self.problem.forward_pass(sample_idx=n)
            loss_n = loss_n * factor
            grad_n = torch.autograd.grad(loss_n, self.problem.model.parameters())

            for param_idx, g_n in enumerate(grad_n):
                batch_grad[param_idx][out_idx] = g_n.detach()

        return batch_grad

    def diag_ggn(self):
        _, output, loss = self.problem.forward_pass()

        def extract_ith_element_of_diag_ggn(i, p):
            v = torch.zeros(p.numel()).to(self.problem.device)
            v[i] = 1.0
            vs = vector_to_parameter_list(v, [p])
            GGN_vs = ggn_vector_product_from_plist(loss, output, [p], vs)
            GGN_v = torch.cat([g.detach().view(-1) for g in GGN_vs])
            return GGN_v[i]

        diag_ggns = []
        for p in list(self.problem.model.parameters()):
            diag_ggn_p = zeros_like(p).view(-1)

            for parameter_index in range(p.numel()):
                diag_value = extract_ith_element_of_diag_ggn(parameter_index, p)
                diag_ggn_p[parameter_index] = diag_value

            diag_ggns.append(diag_ggn_p.view(p.size()))
        return diag_ggns

    def sample_ggn(self, sample_idx=None):
        _, output, loss = self.problem.forward_pass(sample_idx=sample_idx)
        model = self.problem.model

        num_params = sum(p.numel() for p in model.parameters())
        ggn = torch.zeros(num_params, num_params).to(self.problem.device)

        for i in range(num_params):
            # GGN-vector product with i.th unit vector yields the i.th row
            e_i = torch.zeros(num_params).to(self.problem.device)
            e_i[i] = 1.0

            # convert to model parameter shapes
            e_i_list = vector_to_parameter_list(e_i, model.parameters())
            ggn_i_list = ggn_vector_product(loss, output, model, e_i_list)

            ggn_i = parameters_to_vector(ggn_i_list)
            ggn[i, :] = ggn_i

        return ggn

    def ggn(self, subsampling=None):
        if subsampling is None:
            return self.sample_ggn(sample_idx=subsampling)
        else:
            N_axis = 0
            return self.ggn_batch(subsampling=subsampling).sum(N_axis)

    def ggn_batch(self, subsampling=None):
        factor = self.problem.compute_reduction_factor()

        batch_size = self.problem.input.shape[0]
        if subsampling is None:
            subsampling = list(range(batch_size))

        batch_ggn = [None for _ in range(len(subsampling))]

        for out_idx, n in enumerate(subsampling):
            ggn_n = self.sample_ggn(sample_idx=n)
            batch_ggn[out_idx] = factor * ggn_n

        return torch.stack(batch_ggn)

    def diag_ggn_via_ggn(self):
        """Compute full GGN and extract diagonal. Reshape according to param shapes."""
        diag_ggn = self.ggn().diag()

        return vector_to_parameter_list(diag_ggn, self.problem.model.parameters())

    def gammas_ggn(
        self,
        param_groups,
        ggn_subsampling=None,
        grad_subsampling=None,
        directions=False,
    ):
        """First-order derivatives ``γ[n, d]`` along the leading GGN eigenvectors.

        Args:
            param_groups ([dict]): Parameter groups like for ``torch.nn.Optimizer``s.
            ggn_subsampling ([int], optional): Sample indices used for the GGN.
            grad_subsampling ([int], optional): Sample indices used for individual
                gradients.
            directions (bool, optional): Whether to return the directions, too.

        Returns:
            torch.Tensor: 2d tensor containing ``γ[n, d]`` if ``directions=False``.
                Else, a second tensor containing the eigenvectors is returned.
        """
        N, _ = self._mean_reduction()
        _, group_evecs = self.directions_ggn(param_groups, subsampling=ggn_subsampling)

        grad_batch = self.batch_grad(subsampling=grad_subsampling)

        # compensate individual gradient scaling from BackPACK
        individual_gradients = [g * N for g in grad_batch]

        # flattened individual gradients
        individual_gradients = torch.cat(
            [g.flatten(start_dim=1) for g in individual_gradients], dim=1
        )

        indices = parameter_groups_to_idx(
            param_groups, list(self.problem.model.parameters())
        )
        group_igrad = [individual_gradients[:, idx] for idx in indices]

        group_gammas = []

        for igrad, evecs in zip(group_igrad, group_evecs):
            group_gammas.append(torch.einsum("ni,id->nd", igrad, evecs))

        if directions:
            return group_gammas, group_evecs
        else:
            return group_gammas

    def lambdas_ggn(self, param_groups, ggn_subsampling=None, lambda_subsampling=None):
        """Second-order derivatives ``λ[n, d]`` along the leading GGN eigenvectors.

        Uses the exact GGN for λ.

        Args:
            param_groups ([dict]): Parameter groups like for ``torch.nn.Optimizer``s.
            ggn_subsampling ([int], optional): Sample indices used for the GGN.
            lambda_subsampling ([int], optional): Sample indices used for lambdas.

        Returns:
            torch.Tensor: 2d tensor containing ``λ[n, d]``.
        """

        N, _ = self._mean_reduction()
        group_evals, group_evecs = self.directions_ggn(
            param_groups, subsampling=ggn_subsampling
        )

        if lambda_subsampling is None:
            lambda_subsampling = list(range(N))

        group_lambdas = []

        for evals in group_evals:
            D = evals.numel()
            N_lambda = len(lambda_subsampling)

            group_lambdas.append(torch.zeros(N_lambda, D, device=evals.device))

        indices = parameter_groups_to_idx(
            param_groups, list(self.problem.model.parameters())
        )

        for out_idx, n in enumerate(lambda_subsampling):
            ggn_n = self.ggn(subsampling=[n])

            # compensate subsampling scale
            ggn_n *= N

            group_ggn_n = [ggn_n[idx, :][:, idx] for idx in indices]

            for group_idx, (ggn_n, evecs) in enumerate(zip(group_ggn_n, group_evecs)):
                ggn_n_evecs = torch.einsum("ij,jd->id", ggn_n, evecs)
                group_lambdas_n = torch.einsum("id,id->d", evecs, ggn_n_evecs)

                group_lambdas[group_idx][out_idx] = group_lambdas_n

        return group_lambdas

    def directions_ggn(
        self, param_groups, subsampling=None
    ) -> Tuple[List[Tensor], List[Tensor]]:
        """Compute the leading GGN eigenvalues and eigenvectors.

        Args:
            param_groups ([dict]): Parameter groups like for ``torch.nn.Optimizer``s.
            ggn_subsampling ([int], optional): Sample indices used for the GGN.

        Returns:
            First list items are the leading GGN eigenvalues, sorted in ascending order.
            Second tensor are the associated eigenvectors as a column-stacked matrix.
        """
        N, _ = self._mean_reduction()
        ggn = self.ggn(subsampling=subsampling)

        # compensate subsampling scale
        if subsampling is not None:
            ggn *= N / len(subsampling)

        indices = parameter_groups_to_idx(
            param_groups, list(self.problem.model.parameters())
        )
        group_ggn = [ggn[idx, :][:, idx] for idx in indices]

        group_evals = []
        group_evecs = []

        for ggn, group in zip(group_ggn, param_groups):
            evals, evecs = ggn.symeig(eigenvectors=True)

            # select top eigenspace
            criterion = group["criterion"]
            keep = criterion(evals)

            evals = evals[keep]
            self._degeneracy_warning(evals)

            group_evals.append(evals)
            group_evecs.append(evecs[:, keep])

        return group_evals, group_evecs

    def ggn_mat_prod(
        self, mat: List[Tensor], subsampling: List[int] = None
    ) -> List[Tensor]:
        """Multiply each vector in ``mat`` by the GGN.

        Args:
            mat: Stacked vectors in parameter format.
            subsampling: Indices of samples to use for the computation.
                Default: ``None``.

        Returns:
            Stacked results of GGN-vector products in parameter format.
        """
        return self.ggn_mat_prod_from_param_list(
            mat, list(self.problem.model.parameters()), subsampling=subsampling
        )

    def ggn_mat_prod_from_param_list(
        self,
        mat: List[Tensor],
        param_list: List[Parameter],
        subsampling: List[int] = None,
    ) -> List[Tensor]:
        """Multiply each vector in ``mat`` by the GGN w.r.t. the specified parameters.

        Args:
            mat: Stacked vectors in parameter format.
            param_list: List of parameters defining the GGN.
            subsampling: Indices of samples to use for the computation.
                Default: ``None``.

        Returns:
            Stacked results of GGN-vector products in parameter format.
        """

        G_mat = [zeros_like(m) for m in mat]
        num_vecs = G_mat[0].shape[0]

        _, output, loss = self.problem.forward_pass(sample_idx=subsampling)

        for vec_idx in range(num_vecs):
            vec = [m[vec_idx] for m in mat]

            for param_idx, G_v in enumerate(
                ggn_vector_product_from_plist(loss, output, param_list, vec)
            ):
                G_mat[param_idx][vec_idx] = G_v

        return G_mat
