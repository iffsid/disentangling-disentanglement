# regularisers of choice
import torch
from utils import log_mean_exp, kl_divergence

class Regulariser:
    def __init__(self, f, params=None):
        self.f = f
        self.params = params
        self.samples = None
        self.name = None

    def __call__(self, i1, i2):
        raise NotImplementedError


class MMD_DIM(Regulariser):
    def __init__(self):
        super(MMD_DIM, self).__init__(imq_dim_kernel)
        self.samples = True
        self.name = 'mmd_dim'

    def __call__(self, i1, i2):
        """assumes will only be given two inputs (for now)"""
        return self.f(i1, i2)


class KLD_INC(Regulariser):
    def __init__(self):
        super(KLD_INC, self).__init__(kld_inc)
        self.samples = False
        self.name = 'kld'

    def __call__(self, i1, i2):
        """assumes will only be given two inputs (for now)"""
        return self.f(i1, i2)


# functions used to compute regularisers

def kld_inc(pz, qz_x):
    B, D = qz_x.loc.shape
    _zs = pz.rsample(torch.Size([B]))
    lpz = pz.log_prob(_zs).sum(-1).squeeze(-1)
    _zs = _zs.expand(B, B, D)
    lqz = log_mean_exp(qz_x.log_prob(_zs).sum(-1), dim=1)
    inc_kld = lpz - lqz
    inc_kld = inc_kld.mean(0, keepdim=True).expand(1, B)
    return inc_kld.mean(0).sum() / B

def imq_dim_kernel(X, Y):
    assert X.shape == Y.shape
    batch_size, latent_dim = X.shape
    Xb = X.expand(batch_size, *X.shape)
    Yb = Y.expand(batch_size, *Y.shape)
    dists_x = (Xb - Xb.transpose(0, 1)).pow(2)
    dists_y = (Yb - Yb.transpose(0, 1)).pow(2)
    dists_c = (Xb - Yb.transpose(0, 1)).pow(2)
    stats = 0
    off_diag = 1 - torch.eye(batch_size, device=X.device)
    off_diag = off_diag.unsqueeze(-1).expand(*off_diag.shape, latent_dim)
    for scale in [.1, .2, .5, 1., 2., 5.]:
        C = 2 * scale  # 2 * latent_dim * 1.0 * scale
        res1 = C / (C + dists_x)
        res1 += C / (C + dists_y)
        res1 = off_diag * res1
        res1 = res1.sum(0).sum(0) / (batch_size - 1)
        res2 = C / (C + dists_c)
        res2 = res2.sum(0).sum(0) * 2. / (batch_size)
        stats += (res1 - res2).sum()
    return stats / batch_size
