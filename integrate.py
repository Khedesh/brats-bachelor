import sys
import os
import re
import glob
import numpy as np
import SimpleITK as sitk

if __name__ == '__main__':
    index = int(sys.argv[1])
    lf = glob.glob('/tmp/ppr.%d.*.npy' % index)
    print(lf)
    out = 'PR.%d.mha' % index
    ln = []
    for f in lf:
        ln.append(int(re.search(r'(\d+).*?(\d+)', f).group(2)))
    ln = sorted(ln)
    image = np.empty((155, 240, 240))
    for i, n in enumerate(ln):
        if i >= len(ln) - 1:
            image[:, n:, :] = np.load('/tmp/ppr.%d.%d.npy' % (index, n))
            # np.full((155, 240 - n, 240), i + 1)
        else:
            image[:, n:ln[i + 1], :] = np.load('/tmp/ppr.%d.%d.npy' % (index, n))
            # np.full((155, ln[i + 1] - n, 240), i + 1)

        simg = sitk.GetImageFromArray(image)
        sitk.WriteImage(simg, out)
