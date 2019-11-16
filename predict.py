import keras
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
import time


ROWS = 224
COLS = 224
CHANNELS = 3


FORCE_CONVERT = False


def read(name):
    return cv2.imread(name, cv2.IMREAD_COLOR)


def ls(dirname):
    return [path.join(dirname, i) for i in os.listdir(dirname)]


def convert(img):
    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)


def read_img(name):
    img = cv2.imread(name, cv2.IMREAD_COLOR)
    img = convert(img)
    return img


def write_img(img_dir, prediction, output_folder="output"):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    if prediction[0] > prediction[1]:
        label = "dog {:.2%}".format(prediction[0])
    else:
        label = "cat {:.2%}".format(prediction[1])
    img = read_img(img_dir)
    font = cv2.FONT_HERSHEY_SIMPLEX
    topLeftCornerOfText = (10, 30)
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2

    cv2.putText(img, label,
                topLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)
    # Display the image
    # cv2.imshow("img", img)
    # Save image
    cv2.imwrite(path.join(output_folder, os.path.basename(img_dir)), img)

    # cv2.waitKey(0)


def predict(file_name):
    data = np.array([read_img(file_name)])
    model = keras.models.load_model('model.h5')
    model.load_weights('param.hdf5')
    predictions = model.predict(data)
    if predictions[0][0] > predictions[0][1]:
        result = "dog {:.2%}".format(predictions[0][0])
    else:
        result = "cat {:.2%}".format(predictions[0][1])
    print("Prediction result: {}".format(result))
    return result


def predict_folder(folder):
    files = ls(folder)
    data = np.array([read(i) for i in files])
    model = keras.models.load_model('model.h5')
    model.load_weights('param.hdf5')
    predictions = model.predict(data)
    for index, file in enumerate(files, start=0):
        write_img(file, predictions[index])
    result = []
    for i in range(len(predictions)):
        if predictions[i][0] > predictions[i][1]:
            result.append("dog {:.2%}".format(predictions[i][0]))
        else:
            result.append("cat {:.2%}".format(predictions[i][1]))
    return result


if __name__ == "__main__":
    try:
        if len(sys.argv) < 2:
            print(
                "Usage: python predict.py [path of your file]. eg: python predict.py ./test/1.jpg")
        else:
            print("Target file: {}".format(sys.argv[1]))
            result = predict_folder(sys.argv[1])
            print(result)
            print("DONE!")
    except OSError as e:
        print("ERROR: You must run train.py at first!")
