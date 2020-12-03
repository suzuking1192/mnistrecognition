import numpy as np
import os
import keras
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout,Flatten
from keras.optimizers import SGD,Adam
from keras.utils import np_utils

NB_epoch = 20
BATCH_SIZE = 128
VERBOSE =1
NB_classes= 10
OPTIMIZER = Adam()
N_HIDDEN = 128
VALIDATION_SPLIT = 0.2
DROPOUT = 0.3

(X_train,y_train),(X_test,y_test) = mnist.load_data()
image_shape=(28,28,1)


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

X_train=X_train.reshape((-1,28,28,1))
X_test=X_test.reshape((-1,28,28,1))

print(X_train.shape)

y_train = np_utils.to_categorical(y_train,NB_classes)
y_test =np_utils.to_categorical(y_test,NB_classes)

model= Sequential()
model.add(Conv2D(20,kernel_size=5, padding='same',input_shape=image_shape,activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(50,kernel_size=5,padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(500,activation='relu'))
model.add(Dense(NB_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',optimizer=OPTIMIZER,metrics=['accuracy'])

model.fit(X_train,y_train,batch_size=BATCH_SIZE,epochs=NB_epoch,verbose=VERBOSE,validation_split=VALIDATION_SPLIT)

score = model.evaluate(X_test,y_test,verbose = VERBOSE)
print('Test score:',score[0])
print('Test accurasy:',score[1])
