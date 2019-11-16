import os
import sys
import cv2
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns
from os import path
from keras.utils.np_utils import to_categorical

ROWS = 224
COLS = 224
CHANNELS = 3

CACHE_DIR = 'cache/'

# covert data (kể cả khi đã có folder cache)
FORCE_CONVERT = False


def read(name):
    return cv2.imread(name, cv2.IMREAD_COLOR)

def convert(img):
    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)

def save(name, img):
    cv2.imwrite(CACHE_DIR + name, img)
    return img

def ls(dirname):
    return [path.join(dirname,i) for i in os.listdir(dirname)]


def preprocessing(train_dir, test_dir):
    """
    Args::
        - train_dir (str)                           : folder chứa train data
        - test_dir (str)                            : folder chứa test data

    Returns::
        - cache_train_dir (str)                     : folder chứa train data đã covert sang cỡ ảnh 224*224*3
        - cache_test_dir (str)                      : folder chứa test data đã covert sang cỡ ảnh 224*224*3
        - train (np array)                          : ma trận tập data train (dùng cho model)
        - test (np array)                           : ma trận tập data test (dùng cho model)

    Details::
        - convert train and test dataset
    """

    if not os.path.exists(CACHE_DIR):
        os.mkdir(CACHE_DIR)
    if not os.path.exists(path.join(CACHE_DIR, "train")):
        os.mkdir(path.join(CACHE_DIR, "train"))
    if not os.path.exists(path.join(CACHE_DIR, "test")):
        os.mkdir(path.join(CACHE_DIR, "test"))

    cache_train_dir = path.join(CACHE_DIR, "train")
    cache_test_dir = path.join(CACHE_DIR, "test")

    sys.stdout.write('Loading... ')

    train_files = ls(cache_train_dir)
    train = np.array([read(i) for i in train_files])

    test_files = ls(cache_test_dir)
    test = np.array([read(i) for i in test_files])

    print('Done!')

    # if FORCE_CONVERT or len(train) < 25000:
    #     sys.stdout.write('Process train data... ')
    #     train = np.array([save(path.join(train_dir,i), convert(read(path.join(train_dir,i))))
    #                       for i in os.listdir(train_dir)])
    #     train_files = ls(cache_train_dir)
    #     print('Done!')

    # if FORCE_CONVERT or len(test) < 12500:
    #     sys.stdout.write('Process test data... ')
    #     test = np.array([save(path.join(test_dir,i), convert(read(path.join(test_dir,i))))
    #                      for i in os.listdir(test_dir)])
    #     test_files = ls(cache_test_dir)
    #     print('Done!')

    print("Train shape: {}".format(train.shape))
    print("Test shape: {}".format(test.shape))
    return cache_train_dir, cache_test_dir, train, test


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
    train_files = ls(train_files_dir)
    for i in train_files:
        if 'dog' in i:
            labels.append(0)
        else:
            labels.append(1)

    # sns.countplot(labels)
    # plt.title('Dogs and Cats')

    labels = to_categorical(labels)

    return labels


if __name__ == "__main__":
    TRAIN_DIR = 'train/'
    TEST_DIR = 'test/'
    cache_train_dir, cache_test_dir, train, test = preprocessing(
        train_dir=TRAIN_DIR, test_dir=TEST_DIR)
    labels = make_label(train_files_dir=cache_train_dir)
    print("done preprocessing")

