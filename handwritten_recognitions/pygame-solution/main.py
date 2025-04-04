import numpy as np
import matplotlib.pyplot as plt
import keras

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint

(x_train, y_train), (x_test, y_test) = mnist.load_data()

def plot_input_img(i):
    plt.imshow(x_train[i], cmap='binary')
    plt.title(y_train[i])
    plt.axes(False)
    plt.show()

x_train = x_train.astype(np.float32)/255
x_test = x_test.astype(np.float32)/255

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(28,28,1), activation='relu'))
model.add(MaxPool2D((2,2)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPool2D((2,2)))

model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=["accuracy"])

es = EarlyStopping(monitor='val_accuracy', mode='max', min_delta=0.01, patience=4, verbose=1)

mc = ModelCheckpoint('digitrec_model.h5', monitor="val_accuracy", verbose=1, save_best_only=True)

cb = [es,mc]

his = model.fit(x_train, y_train, epochs = 50, validation_split= 0.3, callbacks = cb)

model_S = keras.models.load_model('digitrec_model.h5')

score = model_S.evaluate(x_test, y_test)

print(f"the model accuracy is {score[1]} ")