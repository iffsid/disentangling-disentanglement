import numpy as np
import torch
import math

def compute_disentanglement(zs, ys, L=1000, M=20000):
    '''Metric introduced in Kim and Mnih (2018)'''
    N, D = zs.size()
    _, K = ys.size()
    zs_std = torch.std(zs, dim=0)
    ys_uniq = [c.unique() for c in ys.split(1, dim=1)]  # global: move out
    V = torch.zeros(D, K, device=zs_std.device)
    ks = np.random.randint(0, K, M)      # sample fixed-factor idxs ahead of time

    for m in range(M):
        k = ks[m]
        fk_vals = ys_uniq[k]
        # fix fk
        fk = fk_vals[np.random.choice(len(fk_vals))]
        # choose L random zs that have this fk at factor k
        zsh = zs[ys[:, k] == fk]
        zsh = zsh[torch.randperm(zsh.size(0))][:L]
        d_star = torch.argmin(torch.var(zsh / zs_std, dim=0))
        V[d_star, k] += 1

    return torch.max(V, dim=1)[0].sum() / M

def preprocessed_disentanglement(latents, factors, kls, threshold):
    used_mask = kls > threshold      # threshold
    latents = latents[:, used_mask]  # assumes latents is 2D
    dropdims = (~used_mask).nonzero()
    print('Removing {} latent dimensions: {}'
          .format(len(dropdims), list(dropdims.view(-1).cpu().numpy())))
    return compute_disentanglement(latents, factors)

def compute_sparsity(zs, norm):
    '''
    Hoyer metric
    norm: normalise input along dimension to avoid that dimension collapse leads to good sparsity
    '''
    latent_dim = zs.size(-1)
    if norm:
        zs = zs / zs.std(0)
    l1_l2 = (zs.abs().sum(-1) / zs.pow(2).sum(-1).sqrt()).mean()
    return (math.sqrt(latent_dim) - l1_l2) / (math.sqrt(latent_dim) - 1)
