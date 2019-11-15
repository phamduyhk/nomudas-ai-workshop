import os, sys, cv2, random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns

ROWS = 224
COLS = 224
CHANNELS = 3


FORCE_CONVERT = False


def convert(img):
    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)


def read_img(name):
    img = cv2.imread(name, cv2.IMREAD_COLOR)
    img = convert(img)
    return img

import keras

def predict(file_name):
    data = np.array(read_img(file_name))
    model = keras.models.load_model('model.h5')
    model.load_weights('param.hdf5')
    prediction = model.predict(data)
    print("Prediction: {}".format(prediction))


