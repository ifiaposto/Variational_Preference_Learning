###########################################################################
# we use code found here: https://github.com/VectorInstitute/vbll
###########################################################################

import torch

import warnings


def get_parameterization(p):
    if p in cov_param_dict:
        return cov_param_dict[p]
    else:
        raise ValueError("Must specify a valid covariance parameterization.")


def tp(M):
    return M.transpose(-1, -2)


def sym(M):
    return (M + tp(M)) / 2.0


class Normal(torch.distributions.Normal):
    def __init__(self, loc, chol):
        super(Normal, self).__init__(loc, chol)

    @property
    def mean(self):
        return self.loc

    @property
    def var(self):
        return self.scale**2

    @property
    def chol_covariance(self):
        return torch.diag_embed(self.scale)

    @property
    def covariance_diagonal(self):
        return self.var

    @property
    def covariance(self):
        return torch.diag_embed(self.var)

    @property
    def precision(self):
        return torch.diag_embed(1.0 / self.var)

    @property
    def logdet_covariance(self):
        return 2 * torch.log(self.scale).sum(-1)

    @property
    def logdet_precision(self):
        return -2 * torch.log(self.scale).sum(-1)

    @property
    def trace_covariance(self):
        return self.var.sum(-1)

    @property
    def trace_precision(self):
        return (1.0 / self.var).sum(-1)

    def detach(self):
        return Normal(self.loc.detach(), self.scale.detach())

    def covariance_weighted_inner_prod(self, b, reduce_dim=True):
        assert b.shape[-1] == 1
        prod = (self.var.unsqueeze(-1) * (b**2)).sum(-2)
        return prod.squeeze(-1) if reduce_dim else prod

    def precision_weighted_inner_prod(self, b, reduce_dim=True):
        assert b.shape[-1] == 1
        prod = ((b**2) / self.var.unsqueeze(-1)).sum(-2)
        return prod.squeeze(-1) if reduce_dim else prod

    def __add__(self, inp):
        if isinstance(inp, Normal):
            new_cov = self.var + inp.var
            return Normal(
                self.mean + inp.mean, torch.sqrt(torch.clip(new_cov, min=1e-12))
            )
        elif isinstance(inp, torch.Tensor):
            return Normal(self.mean + inp, self.scale)
        else:
            raise NotImplementedError(
                "Distribution addition only implemented for diag covs"
            )

    def __matmul__(self, inp):
        assert inp.shape[-2] == self.loc.shape[-1]
        assert inp.shape[-1] == 1
        new_cov = self.covariance_weighted_inner_prod(
            inp.unsqueeze(-3), reduce_dim=False
        )
        return Normal(self.loc @ inp, torch.sqrt(torch.clip(new_cov, min=1e-12)))

    def squeeze(self, idx):
        return Normal(self.loc.squeeze(idx), self.scale.squeeze(idx))


class DenseNormal(torch.distributions.MultivariateNormal):
    def __init__(self, loc, cholesky):
        super(DenseNormal, self).__init__(loc, scale_tril=cholesky)

    @property
    def mean(self):
        return self.loc

    @property
    def chol_covariance(self):
        return self.scale_tril

    @property
    def covariance(self):
        return self.scale_tril @ tp(self.scale_tril)

    @property
    def inverse_covariance(self):
        warnings.warn(
            "Direct matrix inverse for dense covariances is O(N^3), consider using eg inverse weighted inner product"
        )
        return tp(torch.linalg.inv(self.scale_tril)) @ torch.linalg.inv(self.scale_tril)

    @property
    def logdet_covariance(self):
        return 2.0 * torch.diagonal(self.scale_tril, dim1=-2, dim2=-1).log().sum(-1)

    @property
    def trace_covariance(self):
        # compute as frob norm squared
        return (self.scale_tril**2).sum(-1).sum(-1)

    def covariance_weighted_inner_prod(self, b, reduce_dim=True):
        assert b.shape[-1] == 1
        prod = ((tp(self.scale_tril) @ b) ** 2).sum(-2)
        return prod.squeeze(-1) if reduce_dim else prod

    def precision_weighted_inner_prod(self, b, reduce_dim=True):
        assert b.shape[-1] == 1
        prod = (torch.linalg.solve(self.scale_tril, b) ** 2).sum(-2)
        return prod.squeeze(-1) if reduce_dim else prod

    def __matmul__(self, inp):
        assert inp.shape[-2] == self.loc.shape[-1]
        assert inp.shape[-1] == 1
        new_cov = self.covariance_weighted_inner_prod(
            inp.unsqueeze(-3), reduce_dim=False
        )
        return Normal(self.loc @ inp, torch.sqrt(torch.clip(new_cov, min=1e-12)))

    def squeeze(self, idx):
        return DenseNormal(self.loc.squeeze(idx), self.scale_tril.squeeze(idx))


class LowRankNormal(torch.distributions.LowRankMultivariateNormal):
    def __init__(self, loc, cov_factor, diag):
        super(LowRankNormal, self).__init__(loc, cov_factor=cov_factor, cov_diag=diag)

    @property
    def mean(self):
        return self.loc

    @property
    def chol_covariance(self):
        raise NotImplementedError()

    @property
    def covariance(self):
        return self.cov_factor @ tp(self.cov_factor) + torch.diag_embed(self.cov_diag)

    @property
    def inverse_covariance(self):
        # TODO(jamesharrison): implement via woodbury
        raise NotImplementedError()

    @property
    def logdet_covariance(self):
        # Apply Matrix determinant lemma
        term1 = torch.log(self.cov_diag).sum(-1)
        arg1 = tp(self.cov_factor) @ (self.cov_factor / self.cov_diag.unsqueeze(-1))
        term2 = torch.linalg.det(arg1 + torch.eye(arg1.shape[-1]).to("cuda")).log()
        return term1 + term2

    @property
    def trace_covariance(self):
        # trace of sum is sum of traces
        trace_diag = self.cov_diag.sum(-1)
        trace_lowrank = (self.cov_factor**2).sum(-1).sum(-1)
        return trace_diag + trace_lowrank

    def covariance_weighted_inner_prod(self, b, reduce_dim=True):
        assert b.shape[-1] == 1
        diag_term = (self.cov_diag.unsqueeze(-1) * (b**2)).sum(-2)
        factor_term = ((tp(self.cov_factor) @ b) ** 2).sum(-2)
        prod = diag_term + factor_term
        return prod.squeeze(-1) if reduce_dim else prod

    def precision_weighted_inner_prod(self, b, reduce_dim=True):
        raise NotImplementedError()

    def __matmul__(self, inp):
        assert inp.shape[-2] == self.loc.shape[-1]
        assert inp.shape[-1] == 1
        new_cov = self.covariance_weighted_inner_prod(
            inp.unsqueeze(-3), reduce_dim=False
        )
        return Normal(self.loc @ inp, torch.sqrt(torch.clip(new_cov, min=1e-12)))

    def detach(self):
        return LowRankNormal(
            self.loc.detach(), self.cov_factor.detach(), self.cov_diag.detach()
        )

    def squeeze(self, idx):
        return LowRankNormal(
            self.loc.squeeze(idx),
            self.cov_factor.squeeze(idx),
            self.cov_diag.squeeze(idx),
        )


cov_param_dict = {"dense": DenseNormal, "diagonal": Normal, "lowrank": LowRankNormal}
