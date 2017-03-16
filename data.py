# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 12:44:13 2017

@author: aslado
"""
from sklearn.model_selection import train_test_split
import numpy as np
import scipy.misc
import sklearn
import csv
import cv2
import skimage.transform as sktransform
print('Loaded data.py with all data processing functions...')
print('...')

# Reading csv file in My Training data and Udacity Training data
# Appending all csv files in one file called samples
samples = []
with open('./my_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

with open('./ud_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# Split data to training and validation data
train_samples, validation_samples = train_test_split(samples, test_size= 0.2)

# Test did split go well
# print('samples combines', len(samples))
# print('train samples all', len(train_samples))
# print('val samples all', len(validation_samples))

def read_image_gray_norma(name):
    # Read image, convert to gray scale and normalize
    image = cv2.imread(name)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = image/255.-.5
    return image

def preprocess(image):
    # Crop image
    image_cropped = image[60:140,0:320]
    # Resize image
    image_cropped = scipy.misc.imresize(image_cropped, (64,64,3)) 
    # cv2.imshow('image',image_cropped)
    # cv2.waitKey(0)
    return image_cropped

def preprocess_flipped_image(image):
    # Flip image 
    flipped_image = cv2.flip(image,1)
    return flipped_image