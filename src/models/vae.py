# Base VAE class definition
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from utils import get_mean_param

class VAE(nn.Module):
    def __init__(self, prior_dist, posterior_dist, likelihood_dist, enc, dec, params):
        super(VAE, self).__init__()
        self.pz = prior_dist
        self.px_z = likelihood_dist
        self.qz_x = posterior_dist
        self.enc = enc
        self.dec = dec
        self.modelName = None
        self.params = params

        self._pz_mu, self._pz_logvar = self.init_pz(params)
        self.prior_variance_scale = params.prior_variance_scale
        self.gamma = nn.Parameter(torch.tensor(params.gamma), requires_grad=False)
        self.df = nn.Parameter(torch.tensor(params.df), requires_grad=False)
        print('p(z):')
        print(self.pz)
        print(self.pz_params)
        print('q(z|x):')
        print(self.qz_x)

        if self.px_z == dist.RelaxedBernoulli:
            self.px_z.log_prob = lambda self, value: \
                -F.binary_cross_entropy_with_logits(
                    self.probs if value.dim() <= self.probs.dim() else self.probs.expand_as(value),
                    value.expand(self.batch_shape) if value.dim() <= self.probs.dim() else value,
                    reduction='none'
                )

    @staticmethod
    def getDataLoaders(batch_size, shuffle=True, device="cuda"):
        raise NotImplementedError

    @property
    def device(self):
        return self._pz_mu.device

    def generate(self, N, K):
        self.eval()
        with torch.no_grad():
            mean_pz = get_mean_param(self.pz_params)
            mean = get_mean_param(self.dec(mean_pz))
            pz = self.pz(*self.pz_params)
            if self.pz == torch.distributions.studentT.StudentT:
                pz._chi2 = torch.distributions.Chi2(pz.df)  # fix from rsample
            px_z_params = self.dec(pz.sample(torch.Size([N])))
            means = get_mean_param(px_z_params)
            samples = self.px_z(*px_z_params).sample(torch.Size([K]))

        return mean, \
            means.view(-1, *means.size()[2:]), \
            samples.view(-1, *samples.size()[3:])

    def reconstruct(self, data):
        self.eval()
        with torch.no_grad():
            qz_x = self.qz_x(*self.enc(data))
            px_z_params = self.dec(qz_x.rsample())

        return get_mean_param(px_z_params)

    def forward(self, x, K=1, no_dec=False):
        qz_x = self.qz_x(*self.enc(x))
        zs = qz_x.rsample(torch.Size([K]))
        if no_dec:
            return qz_x, zs
        px_z = self.px_z(*self.dec(zs))
        return qz_x, px_z, zs

    @property
    def pz_params(self):
        mu = self._pz_mu.mul(1)
        scale = torch.sqrt(self.prior_variance_scale * self._pz_logvar.size(-1) * F.softmax(self._pz_logvar, dim=1))
        return mu, scale

    def init_pz(self, o):
        # means
        pz_mu = nn.Parameter(torch.zeros(1, o.latent_dim), requires_grad=False)

        # variances
        if o.prior_variance == 'iso':
            logvar = torch.zeros(1, o.latent_dim)
        elif o.prior_variance == 'pca':
            singular_values = self.dataset.load_pca(o.latent_dim).log()
            logvar = singular_values.expand(1, o.latent_dim)
        pz_logvar = nn.Parameter(logvar, requires_grad=o.learn_prior_variance)

        return pz_mu, pz_logvar

    def posterior_plot(self, zs_mean, zs_std, runPath, epoch):
        pass
