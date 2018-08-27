import gzip
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from data import BratsDataset as D


def generate_data(start, end):
    brd = D(root_dir)

    data = np.zeros((0, 33, 33))
    label = np.zeros((0,))

    data_test = np.zeros((0, 33, 33))
    label_test = np.zeros((0,))

    for i in range(start, end):
        data_i, label_i = brd[i + 1]
        if data_i.any() and label_i.any():
            print('Data', i, 'preprocess...')
            data_i = data_i.astype(np.float64)
            # z-score normalization
            data_i -= np.mean(data_i)
            data_i /= np.std(data_i)
            data_i = np.pad(data_i, (16,), 'constant', constant_values=(0,))
            # label_i = label_i.reshape((155, 240, 240, 1))
            for x in range(16, 256):
                for y in range(16, 256):
                    print('Patch: ', x, ',', y, 'on', i)
                    if i < 20:
                        data = np.append(data, data_i[x - 16:x + 16, y - 16:y + 16])
                        label = np.append(label, label_i[x - 16, y - 16])
                    else:
                        data_test = np.append(data_test, data_i[x - 16:x + 16, y - 16:y + 16])
                        label_test = np.append(label_test, label_i[x, y])

            print(data.shape, label.shape, data_test.shape, label_test.shape)
            f = gzip.GzipFile('process/data.' + start + '.npy.gz', 'w')
            np.save(f, data)
            f = gzip.GzipFile('process/label.' + start + '.npy.gz', 'w')
            np.save(f, label)
            f = gzip.GzipFile('process/data_test.' + start + '.npy.gz', 'w')
            np.save(f, data_test)
            f = gzip.GzipFile('process/label_test.' + start + '.npy.gz', 'w')
            np.save(f, label_test)


if __name__ == '__main__':
    root_dir = sys.argv[1]

    brd = D(root_dir)

    executor = ProcessPoolExecutor(max_workers=11)

    page = 20
    start = 0
    while start < len(brd):
        print('Start:', start)
        executor.submit(generate_data, start, start + page)
        start += page
