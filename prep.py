import sys

import numpy as np
import gzip

from data import BratsDataset as D

if __name__ == '__main__':
    root_dir = sys.argv[1]

    brd = D(root_dir)

    data = np.zeros((0, 240, 240, 1))
    label = np.zeros((0, 240, 240, 1))

    data_test = np.zeros((0, 240, 240, 1))
    label_test = np.zeros((0, 240, 240, 1))

    for i in range(0, len(brd)):
        data_i, label_i = brd[i + 1]
        if data_i.any() and label_i.any():
            print('Data', i, 'preprocess...')
            data_i = data_i.reshape((155, 240, 240, 1)).astype(np.float64)
            # z-score normalization
            data_i -= np.mean(data_i)
            data_i /= np.std(data_i)
            label_i = label_i.reshape((155, 240, 240, 1))

            if i < 200:
                data = np.append(data, data_i)
                label = np.append(label, label_i)
            else:
                data_test = np.append(data_test, data_i)
                label_test = np.append(label_test, label_i)

    print(data.shape, label.shape, data_test.shape, label_test.shape)
    f = gzip.GzipFile('process/data.npy.gz', 'w')
    np.save(f, data)
    f = gzip.GzipFile('process/label.npy.gz', 'w')
    np.save(f, label)
    f = gzip.GzipFile('process/data_test.npy.gz', 'w')
    np.save(f, data_test)
    f = gzip.GzipFile('process/label_test.npy.gz', 'w')
    np.save(f, label_test)
