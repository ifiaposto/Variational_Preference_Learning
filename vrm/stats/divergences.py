###########################################################################
# we use code found here: https://github.com/VectorInstitute/vbll
###########################################################################
import numpy as np


def gaussian_kl(p, q_var):
    """
    It computes the KL divergence between two Gaussian distributions p and q: KL (p,q).
    p can be any multivariate Gaussian distribution.
    The q distribution has covariance matrix S= q_var* I_N, where I_N is the identity matrix.
    """

    feat_dim = p.mean.shape[-1]

    trace_term = (p.trace_covariance / q_var).sum(-1)
    logdet_term = (feat_dim * np.log(q_var) - p.logdet_covariance).sum(-1)

    mse_term = (p.mean**2).sum(-1).sum(-1) / q_var
    # currently exclude constant
    return 0.5 * (mse_term + trace_term + logdet_term)
