from __future__ import print_function, division

import os
import warnings

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

plt.ion()


class BratsDataset():

    t1_fname = 'T1.mha'
    t1c_fname = 'T1c.mha'
    f_fname = 'F.mha'
    gt_fname = 'GT.mha'
    t2_fname = 'T2.mha'

    def __init__(self, root_dir):
        self.root_dir = root_dir

    def __len__(self):
        _, folders, _ = next(os.walk(self.root_dir))
        return len(folders)

    def get_image(self, index, fname):
        path = os.path.join(self.root_dir, str(index) + '/' + fname)
        if os.path.exists(path):
            img = sitk.ReadImage(path)
            return sitk.GetArrayFromImage(img)
        return np.zeros(1)

    def __getitem__(self, index):
        t1 = self.get_image(index, self.t1_fname)
        t1c = self.get_image(index, self.t1c_fname)
        f = self.get_image(index, self.f_fname)
        t2 = self.get_image(index, self.t2_fname)
        gt = self.get_image(index, self.gt_fname)
        return t1, t1c, f, t2, gt
