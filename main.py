import os, sys, cv2, random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns

np.random.seed(722)

from keras.initializers import TruncatedNormal, Constant
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Input, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, Activation, BatchNormalization
from keras.callbacks import Callback, EarlyStopping
from keras.utils.np_utils import to_categorical

ROWS = 224
COLS = 224
CHANNELS = 3

TRAIN_DIR = 'train/'
TEST_DIR = 'test/'
CACHE_DIR = 'cache/'

FORCE_CONVERT = False


def read(name):
    return cv2.imread(name, cv2.IMREAD_COLOR)

def convert(img):
    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)

def save(name, img):
    cv2.imwrite(CACHE_DIR + name, img)
    return img

def ls(dirname):
    return [dirname + i for i in os.listdir(dirname)]

# 毎回変換していると時間がかかるので、一度変換したらキャッシュします
# キャッシュ用のディレクトリを作ります
if not os.path.exists(CACHE_DIR):
    os.mkdir(CACHE_DIR)
if not os.path.exists(CACHE_DIR + TRAIN_DIR):
    os.mkdir(CACHE_DIR + TRAIN_DIR)
if not os.path.exists(CACHE_DIR + TEST_DIR):
    os.mkdir(CACHE_DIR + TEST_DIR)

sys.stdout.write('Loading... ')

train_files = ls(CACHE_DIR + TRAIN_DIR)
train = np.array([read(i) for i in train_files])

test_files = ls(CACHE_DIR + TEST_DIR)
test = np.array([read(i) for i in test_files])

print('Done!')

if FORCE_CONVERT or len(train) < 25000:
    sys.stdout.write('Process train data... ')
    train =  np.array([save(TRAIN_DIR + i, convert(read(TRAIN_DIR + i))) for i in os.listdir(TRAIN_DIR)])
    train_files = ls(CACHE_DIR + TRAIN_DIR)
    print('Done!')

if FORCE_CONVERT or len(test) < 12500:
    sys.stdout.write('Process test data... ')
    test =  np.array([save(TEST_DIR + i, convert(read(TEST_DIR + i))) for i in os.listdir(TEST_DIR)])
    test_files = ls(CACHE_DIR + TEST_DIR)
    print('Done!')

print("Train shape: {}".format(train.shape))
print("Test shape: {}".format(test.shape))


def make_label(train_files_dir):
    """
        Args::
            - train_files_dir (str)                        : folder chứa data đã covert sang cỡ ảnh 224*224*3

        Returns::
            - labels

        Details::
            - make label with dataset
        """
    labels = []
    for i in train_files_dir:
        if 'dog' in i:
            labels.append(0)
        else:
            labels.append(1)

    # sns.countplot(labels)
    # plt.title('Dogs and Cats')

    labels = to_categorical(labels)

    return labels

labels = make_label(train_files_dir=train_files)

train_dogs = [i for i in train_files if 'dog' in i]
train_cats = [i for i in train_files if 'cat' in i]

def show_train_image(i):
    dog = read(train_dogs[i])
    cat = read(train_cats[i])
    pair = np.concatenate((dog,cat), axis=1)
    plt.figure(figsize=(10,5))
    plt.imshow(pair)
    plt.show()

# for i in range(0,5):
#     show_train_image(i)


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


def run_model():
    model = AlexNet()
    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')
    history = model.fit(train, labels, epochs=1, batch_size=128, shuffle=True, validation_split=0.25, callbacks=[early_stopping])
    return history


def plot_history(history):
    plt.plot(history.history['accuracy'],"o-",label="accuracy")
    plt.plot(history.history['val_accuracy'],"o-",label="val_accuracy")
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


history = run_model()
plot_history(history)
