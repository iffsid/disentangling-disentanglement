# pinwheel model specification

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from .vae import VAE
from utils import Constants
from distributions.normal_mixture import NormalMixture
from .datasets import PinwheelDataset
from vis import scatter_plot, posterior_plot_pinwheel

# Constants
data_size = torch.Size([2])
data_dim = data_size[0]

def extra_hidden_layer(hidden_dim):
    return nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(True))

# Classes
class Enc(nn.Module):
    """ Generate latent parameters for Pinwheel data. """
    def __init__(self, latent_dim, num_hidden_layers=1, hidden_dim=100):
        super(Enc, self).__init__()
        modules = []
        modules.append(nn.Sequential(nn.Linear(data_dim, hidden_dim), nn.ReLU(True)))
        modules.extend([extra_hidden_layer(hidden_dim) for _ in range(num_hidden_layers - 1)])
        self.enc = nn.Sequential(*modules)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        e = self.enc(x.view(*x.size()[:-1], -1))          # flatten data
        return self.fc21(e), F.softplus(self.fc22(e)) + Constants.eta


class Dec(nn.Module):
    """ Generate pinwheel data given a sample from the latent space. """
    def __init__(self, latent_dim, num_hidden_layers=1, hidden_dim=100):
        super(Dec, self).__init__()
        modules = []
        modules.append(nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(True)))
        modules.extend([extra_hidden_layer(hidden_dim) for _ in range(num_hidden_layers - 1)])
        self.dec = nn.Sequential(*modules)
        self.fc31 = nn.Linear(hidden_dim, data_dim)
        self.fc32 = nn.Linear(hidden_dim, data_dim)

    def forward(self, z):
        d = self.dec(z)
        mu = self.fc31(d).view(*z.size()[:-1], data_dim)  # reshape data
        return mu, F.softplus(self.fc32(d)).view_as(mu)

class Pinwheel(VAE):
    """ Derive a specific sub-class of a VAE for Pinwheel. """
    def __init__(self, params):
        assert params.latent_dim == 2
        super(Pinwheel, self).__init__(
            NormalMixture,      # prior
            dist.Normal,        # posterior
            dist.Normal,        # likelihood
            Enc(params.latent_dim, params.num_hidden_layers, params.hidden_dim),
            Dec(params.latent_dim, params.num_hidden_layers, params.hidden_dim),
            params
        )
        self.modelName = 'pinwheel'

    def init_pz(self, o):
        pz_mu = nn.Parameter(torch.tensor([[0., -1.], [0., 1.], [1., 0.], [-1., 0.]]), requires_grad=False)
        pz_logvar = nn.Parameter(torch.zeros(4, o.latent_dim), requires_grad=o.learn_prior_variance)
        return pz_mu, pz_logvar

    @staticmethod
    def getDataLoaders(batch_size, shuffle=True, device="cuda"):
        kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last': True} if device == "cuda" else {'drop_last': True}
        variant = 'small'
        train_loader = DataLoader(
            PinwheelDataset('../data', train=True, variant=variant),
            batch_size=batch_size, shuffle=shuffle, **kwargs)
        test_loader = DataLoader(
            PinwheelDataset('../data', train=False, variant=variant),
            batch_size=batch_size, shuffle=shuffle, **kwargs)
        return train_loader, test_loader

    def generate(self, runPath, epoch):
        N, K = 1000, 100
        mean, means, samples = super(Pinwheel, self).generate(N, K)
        scatter_plot(mean.data.cpu(), '{}/gen_mean_{:03d}.png'.format(runPath, epoch))
        scatter_plot(means.data.cpu(), '{}/gen_means_{:03d}.png'.format(runPath, epoch))
        scatter_plot(samples.data.cpu(),
                     '{}/gen_samples_{:03d}.png'.format(runPath, epoch))

    def reconstruct(self, data, runPath, epoch):
        n = min(400, data.size(0))
        recon = super(Pinwheel, self).reconstruct(data[:n])
        scatter_plot([data[:n].data.cpu(), recon.data.cpu()],
                     '{}/recon_{:03d}.png'.format(runPath, epoch))

    def posterior_plot(self, qz_x_mean, qz_x_std, runPath, epoch):
        posterior_plot_pinwheel(qz_x_mean, qz_x_std, '{}/posterior_{:03d}.png'.format(runPath, epoch))
