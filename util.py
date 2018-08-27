import keras.backend as K


def dice_coef(y_true, y_pred, smooth=1):
    yt = K.flatten(y_true)
    yp = K.flatten(y_pred)
    intersect = K.sum(yt * yp)
    return (2.0 * intersect + smooth) / (K.sum(yt) + K.sum(yp) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1.0-dice_coef(y_true, y_pred)
