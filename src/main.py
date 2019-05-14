import sys
import os
import datetime
import json
import subprocess
import argparse
from tempfile import mkdtemp
from collections import defaultdict

import torch
import numpy as np
from utils import Logger, Timer, save_model, save_vars, probe_infnan
from distributions.sparse import Sparse
import objectives
import regularisers
import models

runId = datetime.datetime.now().isoformat().replace(':', '_')
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Disentangling Disentanglement in VAEs',
                                 formatter_class=argparse.RawTextHelpFormatter)
# General
parser.add_argument('--model', type=str, default='mnist', metavar='M', help='model name (default: mnist)')
parser.add_argument('--name', type=str, default='.', help='experiment name (default: None)')
parser.add_argument('--save-freq', type=int, default=0, help='print objective values every value (if positive)')
parser.add_argument('--skip-test', action='store_true', default=False, help='skip test dataset computations')

# Neural nets
parser.add_argument('--num-hidden-layers', type=int, default=1, metavar='H', help='number of hidden layers in enc and dec (default: 1)')
parser.add_argument('--hidden-dim', type=int, default=100, help='number of units in hidden layers in enc and dec (default: 100)')
parser.add_argument('--fBase', type=int, default=32, help='parameter for DCGAN networks')

# Optimisation
parser.add_argument('--epochs', type=int, default=30, metavar='E', help='number of epochs to train (default: 30)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='batch size for data (default: 64)')

# - Objective
parser.add_argument('--obj', type=str, default='decomp', choices=['decomp', 'iwae'], help='objective to minimise (default: decomp)')
parser.add_argument('--K', type=int, default=1, metavar='K', help='number of samples to estimate ELBO (default: 1)')
parser.add_argument('--beta', type=float, default=1.0, metavar='B', help='overlap factor (default: 1.0)')
parser.add_argument('--alpha', type=float, default=0.0, metavar='A', help='prior regulariser factor (default: 0.0)')
parser.add_argument('--regulariser', type=str, default='kld_inc', metavar='R', choices=['mmd_dim', 'kld_inc'], help='choice of regulariser (default: kld_inc)')

# - Algorithm
parser.add_argument('--beta1', type=float, default=0.9, help='first parameter of Adam (default: 0.9)')
parser.add_argument('--beta2', type=float, default=0.999, help='second parameter of Adam (default: 0.900)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for optimiser (default: 1e-4)')

# Prior / posterior
parser.add_argument('--prior', type=str, default='Normal', help='prior distribution (default: Normal)')
parser.add_argument('--posterior', type=str, default='Normal', help='posterior distribution (default: Normal)')
parser.add_argument('--latent-dim', type=int, default=10, metavar='L', help='latent dimensionality (default: 10)')
parser.add_argument('--gamma', type=float, default=0.8, help='weight of the spike component of the sparse prior')
parser.add_argument('--df', type=float, default=2., help='degree of freedom of the Student-t')

# - weights
parser.add_argument('--prior-variance', type=str, default='iso', choices=['iso', 'pca'], help='value prior variances initialisation')
parser.add_argument('--prior-variance-scale', type=float, default=1., help='scale prior variance by this value (default:1.)')
parser.add_argument('--learn-prior-variance', action='store_true', default=False, help='learn model prior variances')

# Computation
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA use')
parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

# Choosing and saving a random seed for reproducibility
if args.seed == 0:
    args.seed = int(torch.randint(0, 2**32 - 1, (1,)).item())
print('seed', args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True

directory_name = '../experiments/{}'.format(args.name)
if args.name != '.':
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
    runPath = mkdtemp(prefix=runId, dir=directory_name)
else:
    runPath = mkdtemp(prefix=runId, dir=directory_name)
sys.stdout = Logger('{}/run.log'.format(runPath))
print('RunID:', runId)

# save args to run
with open('{}/args.json'.format(runPath), 'w') as fp:
    json.dump(args.__dict__, fp)
with open('{}/args.txt'.format(runPath), 'w') as fp:
    git_hash = subprocess.check_output(['git', 'rev-parse', '--verify', 'HEAD'])
    command = ' '.join(sys.argv[1:])
    fp.write(git_hash.decode('utf-8') + command)
# -- also save object because we want to recover these for other things
torch.save(args, '{}/args.rar'.format(runPath))

modelC = getattr(models, 'VAE_{}'.format(args.model))
train_loader, test_loader = modelC.getDataLoaders(args.batch_size, device=device)
model = modelC(args).to(device)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=args.lr, amsgrad=True, betas=(args.beta1, args.beta2))

objective = getattr(objectives, args.obj + '_objective')
t_objective = getattr(objectives, 'iwae_objective')

regsC = getattr(regularisers, args.regulariser.upper())
regs = regsC()

N = len(train_loader.dataset)
B = args.batch_size
D = args.latent_dim
print('Loss function: ', objective.__name__)
print('Dataset size : ', N)
print('Batch size : ', B)
print('Latent dimensions : ', D)

def train(epoch, beta, alpha, agg):
    model.train()
    b_loss = 0.
    for i, (data, labels) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        loss = -objective(model, data, K=args.K, beta=beta, alpha=alpha,
                          regs=(regs if args.alpha > 0 else None))
        probe_infnan(loss, "Training loss:")
        loss.backward()
        optimizer.step()
        b_loss += loss.item()
        if (args.save_freq == 0 or epoch % args.save_freq == 0) and args.skip_test and i == 0:
            model.reconstruct(data, runPath, epoch)
    agg['train_loss'].append(b_loss / N)
    print('====> Epoch: {:03d} Loss: {:.1f}'.format(epoch, agg['train_loss'][-1]))

@torch.no_grad()
def test(epoch, beta, alpha, agg):
    model.eval()
    b_loss = 0.
    for i, (data, labels) in enumerate(test_loader):
        data = data.to(device)
        loss = -t_objective(model, data, K=args.K, beta=beta, alpha=alpha, regs=None)
        b_loss += loss.item()
        if (args.save_freq == 0 or epoch % args.save_freq == 0) and i == 0:
            model.reconstruct(data, runPath, epoch)
    agg['test_loss'].append(b_loss / len(test_loader.dataset))
    print('====> Test:      Loss: {:.2f}'.format(agg['test_loss'][-1]))

if __name__ == '__main__':
    with Timer('ME-VAE') as t:
        agg = defaultdict(list)
        print('Starting training...')
        for epoch in range(1, args.epochs + 1):
            train(epoch, args.beta, args.alpha, agg)
            save_model(model, runPath + '/model.rar')
            save_vars(agg, runPath + '/losses.rar')
            if (args.save_freq == 0 or epoch % args.save_freq == 0):
                model.generate(runPath, epoch)
            if not args.skip_test:
                test(epoch, args.beta, args.alpha, agg)
        print('p(z) params:')
        print(model.pz_params)
