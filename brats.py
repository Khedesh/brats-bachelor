import os
import gzip

import numpy as np
from keras.callbacks import ModelCheckpoint
from tensorflow.python import debug as tfdbg

from tmi import TMIModel

#
# sess = K.get_session()
# sess = tfdbg.LocalCLIDebugWrapperSession(sess)
# K.set_session(sess)
#

if __name__ == '__main__':
    model = TMIModel().get_model()
    wfile = os.path.join(os.getcwd(), 'weights.h5')
    if os.path.exists(wfile):
        print('Loading Weights...')
        model.load_weights(wfile)
    checkpoint = ModelCheckpoint(wfile, verbose=1, monitor='val_loss', save_best_only=True)

    f = gzip.GzipFile('process/data.npy.gz', 'r')
    data = np.load(f)
    f = gzip.GzipFile('process/label.npy.gz', 'r')
    label = np.load(f)

    model.fit(data, label, epochs=1, verbose=1, batch_size=10, validation_split=0.2, callbacks=[checkpoint])

    f = gzip.GzipFile('process/data_test.npy.gz', 'r')
    data_test = np.load(f)
    f = gzip.GzipFile('process/label_test.npy.gz', 'r')
    label_test = np.load(f)
    print(data_test.shape, label_test.shape)

    # out_test = model.predict(data_test, batch_size=155, verbose=1)
    # score = model.evaluate(data_test, label_test, batch_size=155, verbose=1)
    # print(score)
