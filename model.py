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

samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# Split data to training and validation data
train_samples, validation_samples = train_test_split(samples, test_size= 0.2)

# Test did split go well
print('samples combines', len(samples))
print('train samples all', len(train_samples))
print('val samples all', len(validation_samples))

def read_image_gray_norma(name):
    # Read image, convert to gray scale and normalize
    image = cv2.imread(name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image/255.-.5
    return image

def preprocess(image):
    # Crop image
    image_cropped = image[60:140,0:320]
    # Resize image
    image_cropped = scipy.misc.imresize(image_cropped, (64, 64, 3)) 
    # cv2.imshow('image',image_cropped)
    # cv2.waitKey(0)
    return image_cropped

def preprocess_flipped_image(image):
    # Flip image 
    flipped_image = cv2.flip(image,1)
    return flipped_image

def generator(samples, batch_size=128):
    # Build generator
    num_samples = len(samples)
    while 1:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):                              # With this look we use all images in dataset (Center,Left,Right)
                    source_path = batch_sample[i]
                    split_path = source_path.split('\\')
                    file_name = split_path[-1]
                    local_path = './data/IMG/' + file_name
                    image = preprocess(read_image_gray_norma(local_path))
                    images.append(image)
                correction_angle = 0.18                         # Correction angle due left and right image are looking from different angle
                angle = float(batch_sample[3])
                angles.append(angle)                            # Appending Center Steering angle
                angles.append(angle + correction_angle)         # Appending Left Steering angle
                angles.append(angle - correction_angle)         # Appending Right Steering angle
                
            # flip images and append to directory
            flipped_images = []
            flipped_angles = []
            for image, angle in zip(images, angles):
                flipped_images.append(image)
                flipped_angles.append(angle)
                flipped_image = preprocess_flipped_image(image)
                # flipped_image = cv2.flip(image,1)
                flipped_angle = float(angle) * -1.0
                flipped_images.append(flipped_image)
                flipped_angles.append(flipped_angle)

            #creating array of images and steering angles
            X_train = np.array(flipped_images)
            y_train = np.array(flipped_angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# importing Keras and Keras layers
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Cropping2D
from keras.layers.core import Dropout, Activation
from keras.layers.normalization import BatchNormalization

# size of pooling area for max pooling
pool_size = (2, 2)
subsample = (2, 2)
strides = (1, 1)
activation_relu = 'relu'

#Building model
model = Sequential()
#Add lambda function to avoid pre-processing outside of network
#model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))

# Using Keras Batch Normalization
# Normalize the activations of the previous layer at each batch, 
# i.e. applies a transformation that maintains the mean activation close to 0 
# and the activation standard deviation close to 1.
# model.add(BatchNormalization(epsilon=0.001,mode=2, axis=1,input_shape=(160,320,3)))

# Cropping images
# model.add(Cropping2D(input_shape=(160,320,3), cropping=((60,20),(0,0))))
# NVIDIA model parameters
# model.add(Convolution2D(3, 5, 5, subsample=subsample, activation='relu'))
# model.add(Convolution2D(24, 5, 5, subsample=subsample, activation='relu'))
# model.add(Convolution2D(36, 5, 5, subsample=subsample, activation='relu'))
# model.add(Convolution2D(48, 3, 3, activation='relu'))
# model.add(Convolution2D(64, 3, 3, activation='relu'))
# model.add(Convolution2D(64, 3, 3, activation='relu')) # new one
# model.add(Flatten())
# # model.add(Dense(1164, activation='relu'))
# # model.add(Dropout(.5))
# model.add(Dense(100, activation='relu'))
# model.add(Dropout(.2))
# model.add(Dense(50, activation='relu'))
# model.add(Dropout(.1))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(1, activation='relu'))
# model.summary()
# model.compile(loss='mean_squared_error',
#                    optimizer='adam',
#                    metrics=['accuracy'])

model.add(Convolution2D(16, 3, 3, input_shape=(64, 64, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(500, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(.25))
model.add(Dense(20, activation='relu'))
model.add(Dense(1))
model.summary()
model.compile(loss='mean_squared_error',
                   optimizer='adam',
                   metrics=['accuracy'])

train_generator = generator(train_samples, batch_size=128)
validation_generator = generator(validation_samples, batch_size=128)

#samples_per_epoch = 49488
#nb_val_samples = 12378

# Number of images multiplied 3 cameras and 2 for flipped images (Train #88062, Valid #22020)
samples_per_epoch = len(train_samples)*3*2
nb_val_samples = len(validation_samples)*3*2

model.fit_generator(train_generator, 
                    samples_per_epoch=samples_per_epoch , 
                    validation_data=validation_generator, 
                    nb_val_samples=nb_val_samples,
                    nb_epoch=5)
model.save('model.h5')