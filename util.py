import keras.backend as K
from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects


def dice_coef(y_true, y_pred, smooth=1):
    yt = K.flatten(y_true)
    yp = K.flatten(y_pred)
    intersect = K.sum(yt * yp)
    return K.cast((2 * intersect + smooth) / (K.sum(yt) + K.sum(yp) + smooth), dtype='float32')


def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def MaxProb(x):
    return K.argmax(x, axis=1)


get_custom_objects().update({'MaxProb': Activation(MaxProb)})
