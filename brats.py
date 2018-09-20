import os
import sys

import numpy as np
from keras.callbacks import ModelCheckpoint

from data import BratsDataset
from tmi import TMIModel


# sess = K.get_session()
# sess = tfdbg.LocalCLIDebugWrapperSession(sess)
# K.set_session(sess)


def get_data_and_gt(D, index):
    t1, t1c, f, t2, gt = D[index + 1]
    data = np.stack((t1, t1c, f, t2), axis=3)
    data = np.pad(data, pad_width=((0,), (16,), (16,), (0,)), mode='constant', constant_values=0)
    return data, gt


def data_generator(D, batch_size):
    x, y = 0, 0
    index = 0
    data, gt = get_data_and_gt(D, index)
    print('Data:', data.shape, 'GT:', gt[0:32].shape)
    depth = 155
    axis = 0
    end = False
    while not end:
        remainder = batch_size
        indata = np.empty([0, 33, 33, 4])
        inlabel = np.empty([0])
        while remainder > 0:
            if remainder > depth:
                indata = np.append(indata, data[axis:depth][:, x:x + 33, y:y + 33, :], axis=0)
                inlabel = np.append(inlabel, gt[axis:depth][:, x, y])
                index += 1
                if index == len(D):
                    print('X increment')
                    x += 1
                    if x == 240:
                        x = 0
                        print('Y increment')
                        y += 1
                        if y == 240:
                            end = True
                            break
                    index = 0
                data, gt = get_data_and_gt(D, index)
                indata = np.append(indata, data[0:axis][:, x:x + 33, y:y + 33, :], axis=0)
                inlabel = np.append(inlabel, gt[0:axis][:, x, y])
                remainder -= depth
            elif axis + remainder > depth:
                indata = np.append(indata, data[axis:][:, x:x + 33, y:y + 33, :], axis=0)
                inlabel = np.append(inlabel, gt[axis:][:, x, y])
                index += 1
                if index == len(D):
                    print('X increment')
                    x += 1
                    if x == 240:
                        x = 0
                        print('Y increment')
                        y += 1
                        if y == 240:
                            end = True
                            break
                    index = 0
                data, gt = get_data_and_gt(D, index)
                axis += remainder - depth
                indata = np.append(indata, data[0:axis][:, x:x + 33, y:y + 33, :], axis=0)
                inlabel = np.append(inlabel, gt[0:axis][:, x, y])
                remainder = 0
            else:
                indata = np.append(indata, data[axis:axis + remainder][:, x:x + 33, y:y + 33, :], axis=0)
                inlabel = np.append(inlabel, gt[axis:axis + remainder][:, x, y])
                axis += remainder
                remainder = 0

        yield indata, inlabel


if __name__ == '__main__':
    D = BratsDataset(sys.argv[1])
    print('Number of data: %d' % len(D))
    model = TMIModel().get_model()
    wfile = os.path.join(os.getcwd(), 'weights.h5')
    if os.path.exists(wfile):
        print('Loading Weights...')
        model.load_weights(wfile)
    checkpoint = ModelCheckpoint(wfile, verbose=1, monitor='val_loss', save_best_only=True)
    bs = 256

    try:
        for X, y in data_generator(D, bs):
            print('Generated: ', X.shape, y.shape)
            model.fit(X, y, epochs=1, verbose=1, batch_size=32, callbacks=[checkpoint])
    except StopIteration:
        print('Iteration ended')

        # f = gzip.GzipFile('process/data_test.npy.gz', 'r')
        # data_test = np.load(f)
        # f = gzip.GzipFile('process/label_test.npy.gz', 'r')
        # label_test = np.load(f)
        # print(data_test.shape, label_test.shape)

        # out_test = model.predict(data_test, batch_size=155, verbose=1)
        # score = model.evaluate(data_test, label_test, batch_size=155, verbose=1)
        # print(score)
