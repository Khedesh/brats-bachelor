from keras import Input, Model
from keras.layers import Conv2D, MaxPooling2D, concatenate, Conv2DTranspose
from keras.optimizers import Adam

from model import BaseModel
from util import dice_coef_loss, dice_coef


class UnetModel(BaseModel):
    def __init__(self, base, img_rows, img_cols):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.base = base

    def get_model(self):
        inputs = Input((self.img_rows, self.img_cols, 1))

        conv1 = Conv2D(self.base, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = Conv2D(self.base, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(self.base * 2, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(self.base * 2, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(self.base * 4, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(self.base * 4, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(self.base * 8, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = Conv2D(self.base * 8, (3, 3), activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(self.base * 16, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = Conv2D(self.base * 16, (3, 3), activation='relu', padding='same')(conv5)
        up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)

        conv6 = Conv2D(self.base * 8, (3, 3), activation='relu', padding='same')(up6)
        conv6 = Conv2D(self.base * 8, (3, 3), activation='relu', padding='same')(conv6)
        up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)

        conv7 = Conv2D(self.base * 4, (3, 3), activation='relu', padding='same')(up7)
        conv7 = Conv2D(self.base * 4, (3, 3), activation='relu', padding='same')(conv7)
        up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)

        conv8 = Conv2D(self.base * 2, (3, 3), activation='relu', padding='same')(up8)
        conv8 = Conv2D(self.base * 2, (3, 3), activation='relu', padding='same')(conv8)
        up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)

        conv9 = Conv2D(self.base, (3, 3), activation='relu', padding='same')(up9)
        conv9 = Conv2D(self.base, (3, 3), activation='relu', padding='same')(conv9)
        conv9 = Conv2D(2, (1, 1), activation='relu', padding='same')(conv9)
        conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

        k_model = Model(inputs=[inputs], outputs=[conv10])

        k_model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

        return k_model
