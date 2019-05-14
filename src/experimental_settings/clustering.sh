#!/bin/bash

python3 main.py --name pinwheel_beta_1_alpha_0 --model pinwheel --epochs 500 --save-freq 500 --skip-test --lr 1e-3 --batch-size 400 --latent-dim 2 --prior-variance-scale 0.03 --beta 1. --alpha 0. --regulariser kld_inc
python3 main.py --name pinwheel_beta_0_alpha_1 --model pinwheel --epochs 500 --save-freq 500 --skip-test --lr 1e-3 --batch-size 400 --latent-dim 2 --prior-variance-scale 0.03 --beta 0. --alpha 1. --regulariser kld_inc