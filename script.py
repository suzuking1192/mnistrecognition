import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
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
RESHAPED = 28*28

X_train = X_train.reshape(60000,RESHAPED)
X_test = X_test.reshape(10000,RESHAPED)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

y_train = np_utils.to_categorical(y_train,NB_classes)
y_test =np_utils.to_categorical(y_test,NB_classes)

model= Sequential()
model.add(Dense(N_HIDDEN, input_shape=(RESHAPED,)))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(N_HIDDEN))
model.add(Activation('relu'))
model.add(Dropout(DROPOUT))
model.add(Dense(NB_classes))
model.add(Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',optimizer=OPTIMIZER,metrics=['accuracy'])

model.fit(X_train,y_train,batch_size=BATCH_SIZE,epochs=NB_epoch,verbose=VERBOSE,validation_split=VALIDATION_SPLIT)

score = model.evaluate(X_test,y_test,verbose = VERBOSE)
print('Test score:',score[0])
print('Test accurasy:',score[1])
