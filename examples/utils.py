import tensorflow as tf
import numpy as np
from keras.utils import np_utils


def getImageData(dataSet, num_classes):
    if dataSet == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif dataSet == 'cifar10':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    else:
        raise Exception('Invalid Dataset!')

    if len(x_train.shape) > 3:
        train_channels = 3
    else:
        train_channels = 1

    if len(x_test.shape) > 3:
        test_channels = 3
    else:
        test_channels = 1

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], train_channels)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], test_channels)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    mean = np.mean(x_train, axis=(0, 1, 2, 3))
    std = np.std(x_train, axis=(0, 1, 2, 3))
    x_train = (x_train - mean) / (std + 1e-7)
    x_test = (x_test - mean) / (std + 1e-7)

    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)

    return x_train, y_train, x_test, y_test
