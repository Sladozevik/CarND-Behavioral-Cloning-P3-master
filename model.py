# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 12:44:13 2017

@author: aslado
"""
from sklearn.model_selection import train_test_split
import numpy as np
import sklearn
import csv
import cv2
import skimage.transform as sktransform

# ADD: HOW MANY TIMES DRIVEN AND IN WHICH DIRECTION
# ADD: DID I TRAIN FOR R
# Reading csv file
samples = []
with open('./my_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

with open('./ud_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size= 0.2)

print('samples combines', len(samples))
print('train samples all', len(train_samples))
print('val samples all', len(validation_samples))

def generator(samples, batch_size=64):
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
                    file_name_split = file_name.split('_')
                    if file_name_split[1] == '2016':
                        local_path_ud = './ud_data/IMG/' + '_'.join(file_name_split)
                        image = cv2.imread(local_path_ud)
                    else:
                        local_path_my = './my_data/IMG/' + '_'.join(file_name_split)
                        image = cv2.imread(local_path_my)
                    images.append(image)
                correction_angle = 0.25                         # Correction angle due left and right image are looking from different angle
                angle = float(batch_sample[3])
                angles.append(angle)                            # Appending Center Steering angle
                angles.append(angle + correction_angle)         # Appending Left Steering angle
                angles.append(angle - correction_angle)         # Appending Right Steering angle
                
            # Flip images and append to directory
            flipped_images = []
            flipped_angles = []
            for image, angle in zip(images, angles):
                flipped_images.append(image)
                flipped_angles.append(angle)
                flipped_image = cv2.flip(image,1)
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
model.add(BatchNormalization(epsilon=0.001,mode=2, axis=1,input_shape=(160,320,3)))

# Cropping images
model.add(Cropping2D(cropping=((70,20),(0,0)),input_shape=(160,320,3) ))
model.add(Convolution2D(16, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dropout(.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(.25))
model.add(Dense(20, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

train_generator = generator(train_samples, batch_size=64)
validation_generator = generator(validation_samples, batch_size=64)

samples_per_epoch = 88062
nb_val_samples = 22020

model.fit_generator(train_generator, 
                    samples_per_epoch=samples_per_epoch , 
                    validation_data=validation_generator, 
                    nb_val_samples=nb_val_samples,
                    nb_epoch=5)
model.save('model_test.h5')