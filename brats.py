import os

import numpy as np
import keras.backend as K
from keras.callbacks import ModelCheckpoint
from tensorflow.python import debug as tfdbg

from data import BratsDataset as D
from unet import UnetModel

#
# sess = K.get_session()
# sess = tfdbg.LocalCLIDebugWrapperSession(sess)
# K.set_session(sess)
#

if __name__ == '__main__':
    root_dir = '/Users/Khedesh/Desktop/Workspace/University/Project/Data/BRATS2015_Training/output/HGG'
    model = UnetModel(8, 240, 240).get_model()
    wfile = os.path.join(os.getcwd(), 'weights.h5')
    if os.path.exists(wfile):
        print('Loading Weights...')
        model.load_weights(wfile)
    checkpoint = ModelCheckpoint(wfile, verbose=1, monitor='val_loss', save_best_only=True)

    brd = D(root_dir)

    data = np.zeros((0, 240, 240, 1))
    label = np.zeros((0, 240, 240, 1))
    for i in range(0, 30):
        data_i, label_i = brd[i + 1]
        if data_i.any() and label_i.any():
            data_i = data_i.reshape((155, 240, 240, 1))
            label_i = label_i.reshape((155, 240, 240, 1))
            data = np.append(data, data_i, axis=0)
            label = np.append(label, label_i, axis=0)
    print(data.shape, label.shape)
    model.fit(data, label, epochs=2, verbose=1, batch_size=1, validation_split=0.2, callbacks=[checkpoint])

    data_test, label_test = brd[2]
    data_test = data_test.reshape((155, 240, 240, 1))
    label_test = label_test.reshape((155, 240, 240, 1))
    # score = model.evaluate(data_test, label_test, batch_size=5, verbose=1)
    # print(score)
