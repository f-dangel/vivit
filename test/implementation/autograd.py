from test.implementation.base import ExtensionsImplementation, parameter_groups_to_idx

import torch
from backpack.hessianfree.ggnvp import ggn_vector_product, ggn_vector_product_from_plist
from backpack.hessianfree.rop import R_op
from backpack.utils.convert_parameters import vector_to_parameter_list
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
            loss_n *= factor
            grad_n = torch.autograd.grad(loss_n, self.problem.model.parameters())

            for param_idx, g_n in enumerate(grad_n):
                batch_grad[param_idx][out_idx] = g_n.detach()

        return batch_grad

    def batch_l2_grad(self):
        batch_grad = self.batch_grad()
        batch_l2_grads = [(g ** 2).flatten(start_dim=1).sum(1) for g in batch_grad]
        return batch_l2_grads

    def sgs(self):
        N = self.problem.input.shape[0]
        sgs = [
            torch.zeros(*p.size()).to(self.problem.device)
            for p in self.problem.model.parameters()
        ]

        loss_list = torch.zeros((N))
        gradients_list = []
        for b in range(N):
            _, _, loss = self.problem.forward_pass(sample_idx=b)
            gradients = torch.autograd.grad(loss, self.problem.model.parameters())
            loss_list[b] = loss
            gradients_list.append(gradients)

        _, _, batch_loss = self.problem.forward_pass()
        factor = self.problem.get_reduction_factor(batch_loss, loss_list)

        for _, gradients in zip(range(N), gradients_list):
            for idx, g in enumerate(gradients):
                sgs[idx] += (g.detach() * factor) ** 2
        return sgs

    def variance(self):
        batch_grad = self.batch_grad()
        variances = [torch.var(g, dim=0, unbiased=False) for g in batch_grad]
        return variances

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
            diag_ggn_p = torch.zeros_like(p).view(-1)

            for parameter_index in range(p.numel()):
                diag_value = extract_ith_element_of_diag_ggn(parameter_index, p)
                diag_ggn_p[parameter_index] = diag_value

            diag_ggns.append(diag_ggn_p.view(p.size()))
        return diag_ggns

    def diag_h(self):
        _, _, loss = self.problem.forward_pass()

        def hvp(df_dx, x, v):
            Hv = R_op(df_dx, x, v)
            return [j.detach() for j in Hv]

        def extract_ith_element_of_diag_h(i, p, df_dx):
            v = torch.zeros(p.numel()).to(self.problem.device)
            v[i] = 1.0
            vs = vector_to_parameter_list(v, [p])

            Hvs = hvp(df_dx, [p], vs)
            Hv = torch.cat([g.detach().view(-1) for g in Hvs])

            return Hv[i]

        diag_hs = []
        for p in list(self.problem.model.parameters()):
            diag_h_p = torch.zeros_like(p).view(-1)

            df_dx = torch.autograd.grad(loss, [p], create_graph=True, retain_graph=True)
            for parameter_index in range(p.numel()):
                diag_value = extract_ith_element_of_diag_h(parameter_index, p, df_dx)
                diag_h_p[parameter_index] = diag_value

            diag_hs.append(diag_h_p.view(p.size()))

        return diag_hs

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

    def ggn_mat_prod(self, mat_list, subsampling=None):
        """Vectorized multiplication with the Generalized Gauss-Newton/Fisher.

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
        ggn_mat_list = [None for _ in mat_list]

        for V in range(mat_list[0].shape[0]):
            vec_list = [mat[V] for mat in mat_list]
            ggn_vec_list = self.ggn_vec_prod(vec_list, subsampling=subsampling)
            ggn_vec_list = [ggn_v.unsqueeze(0) for ggn_v in ggn_vec_list]

            # update
            for idx, ggn_v in enumerate(ggn_vec_list):
                ggn_m = ggn_mat_list[idx]
                ggn_m = ggn_v if ggn_m is None else torch.cat([ggn_m, ggn_v])

                ggn_mat_list[idx] = ggn_m

        return ggn_mat_list

    def ggn_vec_prod(self, vec_list, subsampling=None):
        """Multiplication with the Generalized Gauss-Newton/Fisher.

        Args:
            mat_list ([torch.Tensor]): Layer-wise split of vectors to be multiplied by
                the GGN. Each item has the same dimensions as the associated parameter.
            subsampling ([int]): Indices of samples in the mini-batch for which
                the GGN/Fisher should be multiplied with. ``None`` uses the
                entire mini-batch.

        Returns:
            [torch.Tensor]: Result of multiplication with the GGN
        """
        parameters = list(self.problem.model.parameters())

        if subsampling is None:
            _, output, loss = self.problem.forward_pass()
            return ggn_vector_product_from_plist(loss, output, parameters, vec_list)
        else:
            ggn_vec_list = [None for _ in vec_list]
            factor = self.problem.compute_reduction_factor()

            for n in subsampling:
                _, output_n, loss_n = self.problem.forward_pass(sample_idx=n)

                ggn_n_vec_list = [
                    factor * ggn_n_vec
                    for ggn_n_vec in ggn_vector_product_from_plist(
                        loss_n, output_n, parameters, vec_list
                    )
                ]

                # update
                for idx, ggn_n_vec in enumerate(ggn_n_vec_list):
                    old_res = ggn_vec_list[idx]
                    new_res = ggn_n_vec if old_res is None else old_res + ggn_n_vec

                    ggn_vec_list[idx] = new_res

            return ggn_vec_list

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

    def directions_ggn(self, param_groups, subsampling=None):
        """Compute the leading GGN eigenvalues and eigenvectors.

        Args:
            param_groups ([dict]): Parameter groups like for ``torch.nn.Optimizer``s.
            ggn_subsampling ([int], optional): Sample indices used for the GGN.

        Returns:
            (torch.Tensor, torch.Tensor): First tensor are the leading GGN eigenvalues,
                sorted in ascending order. Second tensor are the associated
                eigenvectors as a column-stacked matrix.
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
