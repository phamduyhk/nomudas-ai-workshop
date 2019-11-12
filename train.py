# plot dog photos from the dogs vs cats dataset
from matplotlib import pyplot
from matplotlib.image import imread
from os import listdir
from numpy import asarray
from numpy import save
import os
import numpy as np

# Deep lib
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

# define location of dataset
folder = 'data/train/'


def plot_image(folder):
    # plot first few images
    for i in range(9):
        # define subplot
        pyplot.subplot(330 + 1 + i)
        # define filename
        filename = folder + 'dog.' + str(i) + '.jpg'
        # load image pixels
        image = imread(filename)
        # plot raw pixel data
        pyplot.imshow(image)
    # show the figure
    pyplot.show()
    
def plot_cat_in_train(folder):
    # plot cat photos from the dogs vs cats dataset
    # plot first few images
    for i in range(9):
        # define subplot
        pyplot.subplot(330 + 1 + i)
        # define filename
        filename = folder + 'cat.' + str(i) + '.jpg'
        # load image pixels
        image = imread(filename)
        # plot raw pixel data
        pyplot.imshow(image)
    # show the figure
    pyplot.show()

def load_data(folder):
    # load dogs vs cats dataset, reshape and save to a new file
    photos, labels = list(), list()
    # enumerate files in the directory
    for file in listdir(folder):
        # determine class
        output = 0.0
        if file.startswith('cat'):
            output = 1.0
        # load image
        photo = load_img(folder + file, target_size=(200, 200))
        # convert to numpy array
        photo = img_to_array(photo)
        # store
        photos.append(photo)
        labels.append(output)
    # convert to a numpy arrays
    photos = asarray(photos)
    labels = asarray(labels)
    print(photos.shape, labels.shape)
    # save the reshaped photos
    save('dogs_vs_cats_photos.npy', photos)
    save('dogs_vs_cats_labels.npy', labels)

    # load and confirm the shape
    # photos = np.load('dogs_vs_cats_photos.npy')
    # labels = np.load('dogs_vs_cats_labels.npy')
    # print(photos.shape, labels.shape)

def create_data():
    # create directories
    dataset_home = 'dataset_dogs_vs_cats/'
    subdirs = ['train/', 'test/']
    for subdir in subdirs:
        # create label subdirectories
        labeldirs = ['dogs/', 'cats/']
        for labldir in labeldirs:
            newdir = dataset_home + subdir + labldir
            os.makedirs(newdir, exist_ok=True)
