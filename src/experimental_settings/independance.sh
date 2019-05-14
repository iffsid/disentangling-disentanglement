#!/bin/bash

python3 main.py --name dsprites_beta_1_pz_norm_iso --model dsprites --skip-test --epochs 20 --lr 5e-4 --batch-size 64 --latent-dim 10 --beta 1. --prior-variance iso
python3 main.py --name dsprites_beta_1_pz_norm_pca --model dsprites --skip-test --epochs 20 --lr 5e-4 --batch-size 64 --latent-dim 10 --beta 1. --prior-variance pca
python3 main.py --name dsprites_beta_1_pz_norm_pca_learn --model dsprites --skip-test --epochs 20 --lr 5e-4 --batch-size 64 --latent-dim 10 --beta 1. --prior-variance pca --learn-prior-variance

python3 main.py --name dsprites_beta_1_pz_st_9 --model dsprites --skip-test --epochs 20 --lr 5e-4 --batch-size 64 --latent-dim 10 --beta 1. --prior StudentT --df 9.