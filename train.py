from keras.initializers import TruncatedNormal, Constant
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Input, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, Activation, BatchNormalization
from keras.callbacks import Callback, EarlyStopping
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

import preprocessing

ROWS = 224
COLS = 224
CHANNELS = 3

def conv2d(filters, kernel_size, strides=1, bias_init=1, **kwargs):
    trunc = TruncatedNormal(mean=0.0, stddev=0.01)
    cnst = Constant(value=bias_init)
    return Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding='same',
        activation='relu',
        kernel_initializer=trunc,
        bias_initializer=cnst,
        **kwargs
    )

def dense(units, **kwargs):
    trunc = TruncatedNormal(mean=0.0, stddev=0.01)
    cnst = Constant(value=1)
    return Dense(
        units,
        activation='tanh',
        kernel_initializer=trunc,
        bias_initializer=cnst,
        **kwargs
    )

def AlexNet():
    model = Sequential()

    # 第1畳み込み層
    model.add(conv2d(96, 11, strides=(4,4), bias_init=0, input_shape=(ROWS, COLS, 3)))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization())

    # 第２畳み込み層
    model.add(conv2d(256, 5, bias_init=1))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization())

    # 第３~5畳み込み層
    model.add(conv2d(384, 3, bias_init=0))
    model.add(conv2d(384, 3, bias_init=1))
    model.add(conv2d(256, 3, bias_init=1))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization())

    # 密結合層
    model.add(Flatten())
    model.add(dense(4096))
    model.add(Dropout(0.5))
    model.add(dense(4096))
    model.add(Dropout(0.5))

    # 読み出し層
    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer=SGD(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def run_model(train, labels, epochs=15, batch_size=128, validation_split=0.25):
    model = AlexNet()
    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')
    history = model.fit(train, labels, epochs=epochs, batch_size=batch_size, shuffle=True, validation_split=validation_split, callbacks=[early_stopping])
    return model, history


def plot_history(history):
    plt.plot(history.history['accuracy'],"o-",label="accuracy")
    plt.plot(history.history['val_accuracy'],"o-",label="val_acc")
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

    plt.plot(history.history['loss'],"o-",label="loss",)
    plt.plot(history.history['val_loss'],"o-",label="val_loss")
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.ylim(ymin=0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

if __name__ == "__main__":
    TRAIN_DIR = 'train/'
    TEST_DIR = 'test/'
    cache_train_dir, cache_test_dir, train, test = preprocessing.preprocessing(
        train_dir=TRAIN_DIR, test_dir=TEST_DIR)
    labels = preprocessing.make_label(train_files_dir=cache_train_dir)

    """
    Customize model in here
    epochs=1
    batch_size=32
    validation_split=0.25
    """
    model, history = run_model(train=train,labels=labels,epochs=1,batch_size=32,validation_split=0.25)
    plot_history(history)
    model.save('model.h5')
    model.save_weights('param.hdf5')