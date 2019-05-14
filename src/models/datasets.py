import os
import io
import subprocess
import numpy as np
from scipy.sparse.linalg import svds
from scipy.io import loadmat

import torch
from torchvision import datasets, transforms

from utils import Constants

DspritesDataSize = torch.Size([1, 64, 64])

class DspritesDataset(torch.utils.data.Dataset):
    """2D shapes dataset.
    More info here:
    https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_reloading_example.ipynb
    """
    data_root = '../data'
    filename = 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
    npz_file = data_root + '/' + filename
    npz_train_file = data_root + '/train_' + filename
    npz_test_file = data_root + '/test_' + filename
    pca_filename = data_root + '/pca_' + filename

    def download_dataset(self, npz_file):
        from urllib import request
        url = 'https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz?raw=true'
        print('Downloading ' + url)
        data = request.urlopen(url)
        with open(npz_file, 'wb') as f:
            f.write(data.read())

    def compute_pca(self, path, X):
        print('Compute singular values...')
        _, s, _ = svds(X.reshape(X.shape[0], -1).astype(np.float32), k=20)
        s = s**2
        np.savez(path, pca=s)

    @classmethod
    def load_pca(cls, latent_dim):
        pca = np.load(cls.pca_filename, mmap_mode='r')['pca']
        return torch.Tensor(pca[:latent_dim])

    def split_dataset(self, data_root, npz_file, npz_train_file, npz_test_file, train_fract, clip):
        print('Splitting dataset')
        dataset = np.load(npz_file, encoding='latin1', mmap_mode='r')
        latents = dataset['latents_values'][:, 1:]
        images = np.array(dataset['imgs'], dtype='float32')
        images = images.reshape(-1, *DspritesDataSize)
        if clip:
            images = np.clip(images, Constants.eta, 1 - Constants.eta)

        split_idx = np.int(train_fract * len(latents))
        shuffled_range = np.random.permutation(len(latents))
        train_idx = shuffled_range[range(0, split_idx)]
        test_idx = shuffled_range[range(split_idx, len(latents))]

        np.savez(npz_train_file, images=images[train_idx], latents=latents[train_idx])
        np.savez(npz_test_file, images=images[test_idx], latents=latents[test_idx])

    def __init__(self, data_root, train=True, train_fract=0.8, split=True, clip=False):
        """
        Args:
            npz_file (string): Path to the npz file.
        """
        if not os.path.isfile(self.npz_file):
            self.download_dataset(self.npz_file)
        if split:
            if not (os.path.isfile(self.npz_train_file) and os.path.isfile(self.npz_test_file)):
                self.split_dataset(data_root, self.npz_file, self.npz_train_file,
                                   self.npz_test_file, train_fract, clip)
            dataset = np.load(self.npz_train_file if train else self.npz_test_file,
                              mmap_mode='r')
        else:
            rdataset = np.load(self.npz_file, encoding='latin1', mmap_mode='r')
            dataset = {'latents': rdataset['latents_values'][:, 1:],  # drop colour
                       'images': rdataset['imgs']}

        self.latents = dataset['latents']
        self.images = dataset['images']

    def __len__(self):
        return len(self.latents)

    def __getitem__(self, idx):
        image = torch.Tensor(self.images[idx]).unsqueeze(0)
        latent = torch.Tensor(self.latents[idx])
        return (image, latent)


class PinwheelDataset(torch.utils.data.Dataset):
    """Pinwheel dataset. """

    def __init__(self, dataroot, train=True, variant='boring'):
        """
        Args:
            dataroot (string): Path to the data root directory.
            train (bool): Whether choosing training data or test data.
            variant: One of 'boring', 'rotated', or 'diffarms'.
        """
        self.train = train

        split_name = 'train' if train else 'test'
        datapath = dataroot + '/pinwheel_' + variant + '_' + split_name + '.mat'
        if not os.path.isfile(datapath):
            subprocess.call('../scripts/gen_pinwheel_small.sh', shell=True)

        dataset = loadmat(datapath)

        if self.train:
            self.train_labels = dataset['tY']
            self.train_data = dataset['tX']
        else:
            self.test_labels = dataset['sY']
            self.test_data = dataset['sX']

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __getitem__(self, idx):
        if self.train:
            data, labels = self.train_data[idx], self.train_labels[idx]
        else:
            data, labels = self.test_data[idx], self.test_labels[idx]
        return torch.Tensor(data), torch.Tensor(labels)


def download_dataset(url, file):
    from urllib import request
    print('Downloading ' + file)
    data = request.urlopen(url)
    with open(file, 'wb') as f:
        f.write(data.read())
