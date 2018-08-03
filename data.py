from __future__ import print_function, division

import os
import warnings

import torch

from torch.utils.data import Dataset

import SimpleITK as sitk
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

plt.ion()


class BratsDataset(Dataset):

    gt_fname = 'GT.mha'
    t2_fname = 'T2.mha'

    def __init__(self, root_dir):
        self.root_dir = root_dir

    def __len__(self):
        _, _, files = next(os.walk(self.root_dir))
        return len(files)

    def get_image(self, index, fname):
        img = sitk.ReadImage(os.path.join(self.root_dir, str(index) + '/' + fname))
        return torch.from_numpy(sitk.GetArrayFromImage(img))

    def __getitem__(self, index):
        image = self.get_image(index, self.t2_fname)
        label = self.get_image(index, self.gt_fname)
        return image, label
