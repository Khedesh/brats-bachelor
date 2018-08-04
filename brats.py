import os

import keras.backend as K
from keras.callbacks import ModelCheckpoint
from tensorflow.python import debug as tfdbg

from data import BratsDataset as D
from tmi import TMIModel
from unet import UnetModel
#
# sess = K.get_session()
# sess = tfdbg.LocalCLIDebugWrapperSession(sess)
# K.set_session(sess)
#

if __name__ == '__main__':
    root_dir = '/Users/Khedesh/Desktop/Workspace/University/Project/Data/BRATS2015_Training/output/HGG'
    model = UnetModel(8, 240, 240).get_model()
    data, label = D(root_dir)[1]
    data = data.reshape((155, 240, 240, 1))
    label = label.reshape((155, 240, 240, 1))
    wfile = os.path.join(os.getcwd(), 'weights.h5')
    if os.path.exists(wfile):
        model.load_weights(wfile)
    checkpoint = ModelCheckpoint(wfile, verbose=1, save_best_only=True, period=1)
    print(data.shape, label.shape)
    model.fit(data, label, epochs=10, verbose=1, batch_size=1, callbacks=[checkpoint])

    data_test, label_test = D(root_dir)[2]
    model.evaluate(data_test, label_test, batch_size=5, verbose=1)
