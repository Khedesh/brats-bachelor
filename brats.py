import os
import sys
import gzip

import numpy as np
from keras.callbacks import ModelCheckpoint
from tensorflow.python import debug as tfdbg

from tmi import TMIModel

from data import BratsDataset

import matplotlib.pyplot as plt


#
# sess = K.get_session()
# sess = tfdbg.LocalCLIDebugWrapperSession(sess)
# K.set_session(sess)
#


def data_generator(D, index):
    t1, t1c, f, t2, gt = D[index]
    data = np.stack((t1, t1c, f, t2), axis=1)
    data = np.pad(data, pad_width=((0,), (0,), (16,), (16,)), mode='constant', constant_values=0)
    print(data.shape)
    for r in range(155):
        for x in range(240):
            for y in range(240):
                indata = data[r][:, x:x + 33, y:y + 33]
                inlabel = gt[r][x, y]
                print(indata.shape, inlabel)
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

    for i in range(len(D)):
        model.fit_generator(data_generator(D, i + 1),
                            epochs=1, verbose=1,
                            workers=10, callbacks=[checkpoint])

        # f = gzip.GzipFile('process/data_test.npy.gz', 'r')
        # data_test = np.load(f)
        # f = gzip.GzipFile('process/label_test.npy.gz', 'r')
        # label_test = np.load(f)
        # print(data_test.shape, label_test.shape)

        # out_test = model.predict(data_test, batch_size=155, verbose=1)
        # score = model.evaluate(data_test, label_test, batch_size=155, verbose=1)
        # print(score)
