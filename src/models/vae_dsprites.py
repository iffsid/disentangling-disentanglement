# DSPRITES model specification

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid

import numpy as np
from .vae import VAE
from utils import Constants
from .datasets import DspritesDataset, DspritesDataSize

# Constants
dataSize = DspritesDataSize
imgChans = dataSize[0]

# Classes
class Enc(nn.Module):
    """ Generate latent parameters for dSprites image data. """
    def __init__(self, latent_dim):
        super(Enc, self).__init__()
        self.enc = nn.Sequential(
            # input size: 1 x 64 x 64
            nn.Conv2d(imgChans, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
        )
        self.lin = nn.Linear(64 * 4 * 4, 128)
        self.c1 = nn.Linear(128, latent_dim)
        self.c2 = nn.Linear(128, latent_dim)
        # c1, c2 size: latent_dim x 1 x 1

    def forward(self, x):
        pre_shape = x.size()[:-3]
        e = self.enc(x.view(-1, *dataSize))
        e = e.view(e.shape[0], -1)
        e = self.lin(e)
        mu = self.c1(e).view(*pre_shape, -1)
        return mu, torch.exp(.5 * self.c2(e)).view_as(mu) + Constants.eta

class Dec(nn.Module):
    """ Generate an dSprites image given a sample from the latent space. """
    def __init__(self, latent_dim):
        super(Dec, self).__init__()
        self.lin = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 64 * 4 * 4),
            nn.ReLU(True)
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
        )

    def forward(self, z):
        pre_shape = z.size()[:-1]
        z = self.lin(z)
        z = z.view(-1, 64, 4, 4)
        d = self.dec(z).view(*pre_shape, *dataSize)
        return torch.tensor(1.0).to(z.device), d

class DSPRITES(VAE):
    """ Derive a specific sub-class of a VAE for DSPRITES. """
    def __init__(self, params):
        self.dataset = DspritesDataset
        super(DSPRITES, self).__init__(
            getattr(dist, params.prior),  # prior
            getattr(dist, params.posterior),  # posterior
            dist.RelaxedBernoulli,        # likelihood
            Enc(params.latent_dim),
            Dec(params.latent_dim),
            params
        )
        self.modelName = 'dsprites'

    @property
    def pz_params(self):
        loc = self._pz_mu.mul(1)
        scale = torch.sqrt(self.prior_variance_scale * self._pz_logvar.size(-1) * F.softmax(self._pz_logvar, dim=1))
        if self.pz == dist.StudentT:
            return self.df.mul(1), loc, scale
        return loc, scale

    @staticmethod
    def getDataLoaders(batch_size, shuffle=True, device="cuda"):
        kwargs = {'num_workers': 4, 'pin_memory': True, 'drop_last': False} if device == "cuda" else {}
        print('Load training data...')
        train_loader = DataLoader(DspritesDataset('../data', split=False), batch_size=batch_size, shuffle=shuffle, **kwargs)
        print('Load testing data...')
        test_loader = train_loader
        return train_loader, test_loader

    def generate(self, runPath, epoch):
        N, K = 64, 9
        _, means, samples = super(DSPRITES, self).generate(N, K)
        save_image(means.data.cpu(), '{}/gen_means_{:03d}.png'.format(runPath, epoch))
        # wrangle things so they come out tiled
        samples = samples.view(K, N, *samples.size()[1:]).transpose(0, 1)
        s = [make_grid(t, nrow=int(np.sqrt(K)), padding=0) for t in samples.data.cpu()]
        save_image(torch.stack(s),
                   '{}/gen_samples_{:03d}.png'.format(runPath, epoch),
                   nrow=int(np.sqrt(N)))

    def reconstruct(self, data, runPath, epoch):
        recon = super(DSPRITES, self).reconstruct(data[:8])
        comp = torch.cat([data[:8], recon])
        save_image(comp.data.cpu(), '{}/recon_{:03d}.png'.format(runPath, epoch))
