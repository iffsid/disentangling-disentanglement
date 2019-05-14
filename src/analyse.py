from __future__ import print_function
import argparse
from collections import defaultdict
import os
import torch
import scipy
from utils import Timer
import models
import objectives
import regularisers
from utils import Logger, Timer, save_model, save_vars, probe_infnan
from metrics import compute_disentanglement, compute_sparsity
from vis import plot_latent_magnitude

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Analysing GDVAE results')
parser.add_argument('--save-dir', type=str, metavar='N', help='save directory of results')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA use')
parser.add_argument('--disentanglement', action='store_true', default=False, help='compute disentanglement metric')
parser.add_argument('--sparsity', action='store_true', default=False, help='compute sparsity metric')
parser.add_argument('--logp', action='store_true', default=False, help='estimate tight marginal likelihood on completion')
parser.add_argument('--iwae-samples', type=int, default=1000, help='number of samples for IWAE computation (default: 1000)')

cmds = parser.parse_args()
runPath = cmds.save_dir

args = torch.load(runPath + '/args.rar')

needs_conversion = cmds.no_cuda and args.cuda
conversion_kwargs = {'map_location': lambda st, loc: st} if needs_conversion else {}
args.cuda = not cmds.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

modelC = getattr(models, 'VAE_{}'.format(args.model))
model = modelC(args)
if args.cuda:
    model.cuda()

state_dict = torch.load(runPath + '/model.rar', **conversion_kwargs)
model.load_state_dict(state_dict)
train_loader, test_loader = model.getDataLoaders(args.batch_size, device=device)

objective = getattr(objectives, args.obj + '_objective')
regsC = getattr(regularisers, args.regulariser.upper())
regs = regsC()

N = len(test_loader.dataset)
B = args.batch_size
D = args.latent_dim

@torch.no_grad()
def test(beta, alpha, agg):
    model.eval()
    b_negloss, b_recon, b_kl, b_reg, b_mlike = 0., 0., 0., 0., 0.
    zs_mean = torch.zeros(len(test_loader.dataset), D, device=device)
    zs_std = torch.zeros(len(test_loader.dataset), D, device=device)
    zs2_mean = torch.zeros(len(test_loader.dataset), D, device=device)
    L = test_loader.dataset[0][1].view(-1).size(-1)
    ys = torch.zeros(len(test_loader.dataset), L, device=device)
    for i, (data, labels) in enumerate(test_loader):
        data = data.to(device)
        qz_x, px_z, zs = model(data, 1)
        negloss, recon, kl, reg = objective(model, data, K=args.K, beta=beta, alpha=alpha,
                                            regs=(regs if args.alpha > 0 else None), components=True)
        b_negloss += negloss.item()
        b_recon += recon.item()
        b_kl += kl.item()
        b_reg += reg.item()

        zs_mean[(B * i):(B * (i + 1)), :] = qz_x.mean
        zs_std[(B * i):(B * (i + 1)), :] = qz_x.stddev
        ys[(B * i):(B * (i + 1)), :] = labels.view(-1, L)

        if cmds.disentanglement:
            # change measure if prior is not normal (along optimal transport map)
            if model.pz == torch.distributions.studentT.StudentT:
                df, pz_mean, pz_scale = model.pz_params
                u = scipy.stats.t.cdf(qz_x.mean.data.cpu().numpy(), df=df.data.cpu().numpy(),
                                      loc=pz_mean.data.cpu().numpy(), scale=pz_scale.data.cpu().numpy())
                qz_x_mean = torch.distributions.Normal(loc=pz_mean, scale=pz_scale).icdf(torch.tensor(u, dtype=torch.float).to(device))
            else:
                qz_x_mean = qz_x.mean
            zs2_mean[(B * i):(B * (i + 1)), :] = qz_x_mean.view(B, D)

        if cmds.logp:
            b_mlike += objectives.iwae_objective(model, data, cmds.iwae_samples).sum().item()

    agg['test_loss'].append(-b_negloss / N)
    agg['test_recon'].append(b_recon / N)
    agg['test_kl'].append(b_kl / N)
    agg['test_reg'].append(b_reg / N)
    print('Loss: {:.1f} Recon: {:.1f} KL: {:.1f} Reg: {:.3f}'
          .format(agg['test_loss'][-1], agg['test_recon'][-1], agg['test_kl'][-1], agg['test_reg'][-1]))

    model.posterior_plot(zs_mean, zs_std, runPath, args.epochs)

    if cmds.disentanglement:
        dis = compute_disentanglement(zs_mean, ys).item()
        agg['test_disentanglement'].append(dis)
        dis2 = compute_disentanglement(zs2_mean, ys).item()
        agg['test_disentanglement2'].append(dis2)
        print('Disentanglement: {:.3f} (wo OT {:.3f})'.format(agg['test_disentanglement2'][-1], agg['test_disentanglement'][-1]))

    if cmds.sparsity:
        agg['test_sparsity'].append(compute_sparsity(zs_mean, norm=True))
        print('Sparsity: {:.3f}'.format(agg['test_sparsity'][-1]))
        labels = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        C = len(ys.unique())
        zs_mean_mag_avg = torch.zeros(C, zs_mean.size(-1))
        for c in sorted(list(ys.unique())):
            idx = (ys == c).view(-1)
            zs_mean_mag_avg[int(c)] = zs_mean[idx].abs().mean(0)
        plot_latent_magnitude(zs_mean_mag_avg[range(10), :], labels=labels, path=runPath + '/plot_sparsity')

    if cmds.logp:
        agg['test_mlik'].append(b_mlike / len(test_loader.dataset))
        print('Marginal Log Likelihood (IWAE, K = {}): {:.4f}'.format(cmds.iwae_samples, agg['test_mlik'][-1]))

agg = defaultdict(list)
test(args.beta, args.alpha, agg)
save_vars(agg, runPath + '/losses2.rar')
