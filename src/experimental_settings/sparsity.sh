#!/bin/bash

python3 main.py --name fashion_mnist_gamma_0.0_alpha_0 --model fashion_mnist --skip-test --epochs 80 --save-freq 80 --batch-size 500 --lr 5e-4 --latent-dim 50 --regulariser mmd_dim --beta 1. --alpha 0. --gamma 0.0
python3 main.py --name fashion_mnist_gamma_0.0_alpha_100 --model fashion_mnist --skip-test --epochs 80 --save-freq 80 --batch-size 500 --lr 5e-4 --latent-dim 50 --regulariser mmd_dim --beta 1. --alpha 100. --gamma 0.0
python3 main.py --name fashion_mnist_gamma_0.8_alpha_0 --model fashion_mnist --skip-test --epochs 80 --save-freq 80 --batch-size 500 --lr 5e-4 --latent-dim 50 --regulariser mmd_dim --beta 1. --alpha 0. --gamma 0.8
python3 main.py --name fashion_mnist_gamma_0.8_alpha_100 --model fashion_mnist --skip-test --epochs 80 --save-freq 80 --batch-size 500 --lr 5e-4 --latent-dim 50 --regulariser mmd_dim --beta 1. --alpha 100. --gamma 0.8