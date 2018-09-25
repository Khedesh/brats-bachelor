import os
import sys

import numpy as np
from keras.callbacks import ModelCheckpoint
import SimpleITK as sitk

import config
from data import BratsDataset
from tmi import TMIModel

import tensorflow as tf
from keras.backend import tensorflow_backend as K


# sess = K.get_session()
# sess = tfdbg.LocalCLIDebugWrapperSession(sess)
# K.set_session(sess)


def get_data_and_gt(D, index):
    t1, t1c, f, t2, gt = D[index + 1]
    data = np.stack((t1, t1c, f, t2), axis=3)
    data = np.pad(data, pad_width=((0,), (16,), (16,), (0,)), mode='constant', constant_values=0)
    return data, gt


def categoricalize(label):
    ret = np.zeros((label.shape[0], 5), dtype='int8')
    for i in range(label.shape[0]):
        ret[i][label.astype(dtype='int8')[i]] = 1
    return ret


def decategoricalize(output):
    ret = np.empty(output.shape[:-1])
    with np.nditer(ret, op_flags=['readwrite']) as it:
        for x in it:
            x[...] = np.argmax(output[...])
    return ret


def data_generator(D, batch_size,
                   min_index=0, max_index=-1,
                   min_x=0, max_x=239,
                   min_y=0, max_y=239):
    x, y = min_x, min_y
    index = min_index
    max_index = len(D) if max_index == -1 else max_index
    data, gt = get_data_and_gt(D, index)
    print('Data:', data.shape, 'GT:', gt[0:32].shape)
    depth = 155
    axis = 0
    end = False
    while not end:
        remainder = batch_size
        indata = np.empty([0, 33, 33, 4])
        inlabel = np.empty([0])
        while remainder > 0:
            if remainder > depth:
                indata = np.append(indata, data[axis:depth][:, x:x + 33, y:y + 33, :], axis=0)
                inlabel = np.append(inlabel, gt[axis:depth][:, x, y])
                index += 1
                if index > max_index:
                    print('X, Y:', x, y)
                    print('X increment:', x)
                    x += 1
                    if x > max_x:
                        x = min_x
                        print('Y increment:', y)
                        y += 1
                        if y > max_y:
                            end = True
                            break
                    index = min_index
                data, gt = get_data_and_gt(D, index)
                indata = np.append(indata, data[0:axis][:, x:x + 33, y:y + 33, :], axis=0)
                inlabel = np.append(inlabel, gt[0:axis][:, x, y])
                remainder -= depth
            elif axis + remainder > depth:
                indata = np.append(indata, data[axis:][:, x:x + 33, y:y + 33, :], axis=0)
                inlabel = np.append(inlabel, gt[axis:][:, x, y])
                index += 1
                if index > max_index:
                    print('X, Y:', x, y)
                    print('X increment:', x)
                    x += 1
                    if x > max_x:
                        x = min_x
                        print('Y increment:', y)
                        y += 1
                        if y > max_y:
                            end = True
                            break
                    index = min_index
                data, gt = get_data_and_gt(D, index)
                axis += remainder - depth
                indata = np.append(indata, data[0:axis][:, x:x + 33, y:y + 33, :], axis=0)
                inlabel = np.append(inlabel, gt[0:axis][:, x, y])
                remainder = 0
            else:
                indata = np.append(indata, data[axis:axis + remainder][:, x:x + 33, y:y + 33, :], axis=0)
                inlabel = np.append(inlabel, gt[axis:axis + remainder][:, x, y])
                axis += remainder
                remainder = 0

        yield indata, categoricalize(inlabel)


if __name__ == '__main__':
    with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=16)) as sess:
        K.set_session(sess)

        D = BratsDataset(config.DATA_CONFIG['root_dir'])
        print('Number of data: %d' % len(D))
        model = TMIModel().get_model()
        wfile = os.path.join(os.getcwd(), config.APP_CONFIG['weights_file'])
        if os.path.exists(wfile):
            print('Loading Weights...')
            model.load_weights(wfile)
        checkpoint = ModelCheckpoint(wfile, verbose=1, monitor='val_loss', save_best_only=True, period=1)
        bs = config.DATA_CONFIG['batch_size']

        if config.APP_CONFIG['train']:
            print('Training...')
            try:
                for X, y in data_generator(D, bs, min_index=config.DATA_CONFIG['min_train_index'],
                                           max_index=config.DATA_CONFIG['max_train_index']):
                    print('Generated for train: ', X.shape, y.shape)
                    model.fit(X, y, epochs=config.TRAIN_CONFIG['epochs'], verbose=1,
                              batch_size=config.TRAIN_CONFIG['batch_size'],
                              validation_split=0.25, callbacks=[checkpoint])
            except StopIteration:
                print('Training iteration ended')

        if config.APP_CONFIG['test']:
            print('Testing...')
            print('Metrics:', model.metrics_names)
            try:
                for X, y in data_generator(D, bs, min_index=config.DATA_CONFIG['min_test_index'],
                                           max_index=config.DATA_CONFIG['max_test_index']):
                    print('Generated for test:', X.shape, y.shape)
                    score = model.evaluate(X, y, verbose=1, batch_size=config.TEST_CONFIG['batch_size'])
                    print('Score:', score)
            except StopIteration:
                print('Test iteration ended')

        if config.APP_CONFIG['predict']:
            print('Predicting...')
            for i in range(config.DATA_CONFIG['min_predict_index'],
                           config.DATA_CONFIG['max_predict_index']):
                try:
                    x, y = 0, 0
                    dx = int(sys.argv[2]) - int(sys.argv[1]) + 1
                    image = np.empty((155, dx, 240))
                    for X, _ in data_generator(D, config.PREDICT_CONFIG['batch_size'],
                                               min_index=i, max_index=i,
                                               min_x=int(sys.argv[1]), max_x=int(sys.argv[2])):
                        print('Generated for predict:', X.shape)
                        output = model.predict(X, verbose=1,
                                               batch_size=config.PREDICT_CONFIG['batch_size'])
                        output = decategoricalize(output)
                        image[:, x, y] = output
                        x += 1
                        if x >= dx:
                            y += 1
                            x = 0
                            if y == 240:
                                break
                        # simg = sitk.GetImageFromArray(image)
                        # sitk.WriteImage(simg, 'PR.' + str(i) + '.mha')
                    np.save('/tmp/ppr.%d.%d.npy' % (i, x), image)
                except StopIteration:
                    print('Predict iteration ended')
