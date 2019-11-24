import keras
from keras import Sequential
from keras.engine import InputLayer
from keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

from examples.custom_optimiser import LRMultiplierSGD
from examples.custom_weight_decay import add_weight_decay
from examples.utils import getImageData

num_classes = 10
x_train, y_train, x_test, y_test = getImageData('cifar10', num_classes)

shape = x_train.shape[1:]

kernel_init = keras.initializers.RandomNormal(stddev=0.05)

model = Sequential()
model.add(InputLayer(input_shape=(32, 32, 3)))
# Unit 1
model.add(Conv2D(filters=32, kernel_size=(5, 5), kernel_initializer=kernel_init))
model.add(MaxPooling2D(padding='same'))
model.add(Activation('relu'))
# Unit 2
model.add(Conv2D(filters=32, kernel_size=(5, 5), kernel_initializer=kernel_init))
model.add(MaxPooling2D(padding='same'))
model.add(Activation('relu'))
# Unit 3
model.add(Conv2D(filters=32, kernel_size=(5, 5), kernel_initializer=kernel_init))
model.add(MaxPooling2D(padding='same'))
model.add(Activation('relu'))
# Output
model.add(Flatten())
model.add(Dense(units=10))
model.add(Activation('softmax'))

add_weight_decay(model, 0.0005)
opt = SGD(lr=0.01, momentum=0., decay=0.0)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    zca_epsilon=1e-06,  # epsilon for ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    # randomly shift images horizontally (fraction of total width)
    width_shift_range=0.1,
    # randomly shift images vertically (fraction of total height)
    height_shift_range=0.1,
    shear_range=0.,  # set range for random shear
    zoom_range=0.,  # set range for random zoom
    channel_shift_range=0.,  # set range for random channel shifts
    # set mode for filling points outside the input boundaries
    fill_mode='nearest',
    cval=0.,  # value used for fill_mode = "constant"
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False,  # randomly flip images
    # set rescaling factor (applied before any other transformation)
    rescale=None,
    # set function that will be applied on each input
    preprocessing_function=None,
    # image data format, either "channels_first" or "channels_last"
    data_format=None,
    # fraction of images reserved for validation (strictly between 0 and 1)
    validation_split=0.0)

datagen.fit(x_train)

model.fit_generator(datagen.flow(x_train, y_train, batch_size=128), epochs=1,
                    validation_data=(x_test, y_test),
                    workers=4)

loss, acc = model.evaluate(x_test, y_test)
print(f'Accuracy: {acc * 100}%')
print(f'Error: {(1 - acc) * 100}%')


# datagen = ImageDataGenerator(
#     featurewise_center=False,  # set input mean to 0 over the dataset
#     samplewise_center=False,  # set each sample mean to 0
#     featurewise_std_normalization=False,  # divide inputs by std of the dataset
#     samplewise_std_normalization=False,  # divide each input by its std
#     zca_whitening=False,  # apply ZCA whitening
#     zca_epsilon=1e-06,  # epsilon for ZCA whitening
#     rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
#     # randomly shift images horizontally (fraction of total width)
#     width_shift_range=0.,
#     # randomly shift images vertically (fraction of total height)
#     height_shift_range=0.,
#     shear_range=0.,  # set range for random shear
#     zoom_range=0.,  # set range for random zoom
#     channel_shift_range=0.,  # set range for random channel shifts
#     # set mode for filling points outside the input boundaries
#     fill_mode='nearest',
#     cval=0.,  # value used for fill_mode = "constant"
#     horizontal_flip=False,  # randomly flip images
#     vertical_flip=False,  # randomly flip images
#     # set rescaling factor (applied before any other transformation)
#     rescale=None,
#     # set function that will be applied on each input
#     preprocessing_function=None,
#     # image data format, either "channels_first" or "channels_last"
#     data_format=None,
#     # fraction of images reserved for validation (strictly between 0 and 1)
#     validation_split=0.0)