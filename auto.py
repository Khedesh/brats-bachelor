import os

from keras.callbacks import ModelCheckpoint

import config
from data import BratsDataset
from unet import UnetModel

if __name__ == '__main__':
    model = UnetModel(64, 240, 240).get_model()
    D = BratsDataset(config.DATA_CONFIG['root_dir'])
    wfile = os.path.join(os.getcwd(), config.APP_CONFIG['weights_file'])
    if os.path.exists(wfile):
        print('Loading Weights...')
        model.load_weights(wfile)
    checkpoint = ModelCheckpoint(wfile, verbose=1, monitor='val_loss', save_best_only=True, period=1)

    if config.APP_CONFIG['train']:
        print('Training...')
        for i in range(len(D)):
            print(D[i + 1][0].shape)
            for j in range(155):
                X = D[i + 1][0].reshape((155, 240, 240, 1))
                model.fit(X, D[i + 1][4],
                          batch_size=32, epochs=10,
                          verbose=1, validation_split=.2,
                          callbacks=[checkpoint])
