import keras
from keras import Sequential
from keras.engine import InputLayer
from keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense
from keras.optimizers import SGD

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

model.fit(x_train, y_train, batch_size=128, epochs=100)

loss, acc = model.evaluate(x_test, y_test)
print(f'Accuracy: {acc * 100}%')
print(f'Error: {(1- acc) * 100}%')
