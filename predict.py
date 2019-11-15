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
    if prediction is 0:
        result = "dog"
    else:
        result = "cat"
    print("Prediction result: {}".format(result))
    return result
    
    
if __name__ == "__main__":
    try:
        if len(sys.argv)<2:
            print("Usage: python predict.py [path of your file]. eg: python predict.py ./test/1.jpg")
        else:
            print("Target file: {}".format(sys.argv[1]))
            result = predict(sys.argv[1])
            I = cv2.imread(sys.argv[1])
            cv2.namedWindow(result)
            cv2.imshow(result, I)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    except OSError as e:
        print("ERROR: You must run train.py at first!")


