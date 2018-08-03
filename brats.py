import keras.backend as K
from keras.layers import Conv2D, MaxPooling2D, Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint

from data import BratsDataset as D

import os

def dice_coef(y_true, y_pred, smooth=1):
    yt = K.flatten(y_true)
    yp = K.flatten(y_pred)
    intersect = K.sum(yt * yp)
    return (2.0 * intersect + smooth) / (K.sum(yt) + K.sum(yp) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_model():
    model = Sequential()

    model.add(
        Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_last', activation='relu'))
    model.add(
        Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_last', activation='relu'))
    model.add(
        Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_last', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', data_format='channels_last'))

    model.add(
        Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_last', activation='relu'))
    model.add(
        Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_last', activation='relu'))
    model.add(
        Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_last', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', data_format='channels_last'))

    model.add(Dense(256, input_shape=(6272,), activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(2, activation='relu'))

    sgd = SGD(lr=0.01, momentum=0.5, nesterov=True)
    model.compile(optimizer=sgd,
                  loss=dice_coef_loss,
                  metrics=[dice_coef])
    return model


if __name__ == '__main__':
    root_dir = '/Users/Khedesh/Desktop/Workspace/University/Project/Data/BRATS2015_Training/output/HGG'
    model = get_model()
    data, label = D(root_dir)[1]
    data = data.reshape((155, 240, 240, 1))
    label = label.reshape((155, 240, 240, 1))
    wfile = 'weights.h5'
    if os.path.exists(wfile):
        model.load_weights(wfile)
    checkpoint = ModelCheckpoint(wfile, verbose=1, save_best_only=True)
    print(data.shape, label.shape)
    model.fit(data, label, epochs=10, verbose=1, batch_size=1, callbacks=[checkpoint])

