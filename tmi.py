from keras.layers import Conv2D, MaxPooling2D, Dense, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import SGD

from model import BaseModel
from util import dice_coef_loss, dice_coef, MaxProb


class TMIModel(BaseModel):
    def get_model(self):
        model = Sequential()

        model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_last',
                         activation='relu', input_shape=(33, 33, 4)))
        model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_last',
                         activation='relu'))
        model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_last',
                         activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', data_format='channels_last'))

        model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_last',
                         activation='relu'))
        model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_last',
                         activation='relu'))
        model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', data_format='channels_last',
                         activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', data_format='channels_last'))

        model.add(Flatten())
        model.add(Dense(256, input_shape=(6272,), activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(5))
        model.add(Activation('softmax'))
        model.add(Activation(MaxProb))

        sgd = SGD(lr=0.01, momentum=0.5, nesterov=True)
        model.compile(optimizer=sgd,
                      loss=dice_coef_loss,
                      metrics=[dice_coef])
        return model
