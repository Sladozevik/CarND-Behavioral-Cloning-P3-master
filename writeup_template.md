#**Udacity Self Driving Car - Behavioral Cloning** 

##Ante Sladoejvic - Writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image2]: ./examples/img_driving.png "Image Driving"
[image3]: ./examples/img_rgb.png "Image RGB"
[image4]: ./examples/img_gray.png "Image Gray"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

###1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

###2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

###3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

###1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 16 and 64 (model.py lines 138-157) 

The model includes RELU layers to introduce nonlinearity (code lines 138-157), and the data is normalized in the model using a within python function read_image_gray_norma (code line 29). 


###2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 149,151,153). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

###3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 157).

###4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and made around 6 laps in one direction and 3 laps in oposite direction

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

###1. Solution Design Approach

My first step was to use a convolution neural network model similar to the Nvidia research paper I thought this model might be appropriate because it worked.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that Nvidia model had a low mean squared error and did not succeed to drive whole lap. On the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model by copying Nvidia model, adding dropout. After that i added MaxPooling layers.

Then I flattened model from 1000 -> 500 -> 100 -> 20 -> 1

The final step was to run the simulator to see how well the car was driving around track one. 
While i was improving my model vehicle was constantly felling off the track. I was using each drive to improve model. From fine tuning model to adding cropping, resizing image and graying out images.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

###2. Final Model Architecture

The final model architecture (model.py lines 138-157) consisted of a convolution neural network with the following layers and layer sizes:

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

Here is summary of how network looks:

(Convolution2D (16, 3, 3))  (Output Shape: 62, 62, 16)

(MaxPooling2D)    			(Output Shape: 31, 31, 16)

(Convolution2D (32, 3, 3))  (Output Shape: 29, 29, 32)

(MaxPooling2D)  			(Output Shape: 14, 14, 32)

(Convolution2D (32, 3, 3))  (Output Shape: 12, 12, 32)

(Convolution2D (64, 3, 3))  (Output Shape: 10, 10, 64)

(MaxPooling2D)  			(Output Shape: 5, 5, 64)  

(Convolution2D (64, 3, 3))  (Output Shape: 3, 3, 64)  

(MaxPooling2D)  			(Output Shape: 1, 1, 64)  

flatten_1 (Flatten)         (Output Shape: 64)    

dense_1 (Dense)             (Output Shape: 1000)  

dropout_1 (Dropout)         (Output Shape: 1000)  

dense_2 (Dense)             (Output Shape: 500)   

dropout_2 (Dropout)         (Output Shape: 500)   

dense_3 (Dense)             (Output Shape: 100)   

dropout_3 (Dropout)         (Output Shape: 100)   

dense_4 (Dense)             (Output Shape: 20)    

dense_5 (Dense)             (Output Shape: 1)     

Total params: 687,401

Trainable params: 687,401

Non-trainable params: 0

Epoch 1/5

###3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded 6 laps on track one using center lane driving after that i driven in opposite direction for 3 laps. Here is an example image of center lane driving after cropping:

![alt text][image2]

To augment the data set, I also flipped images and angles thinking that this would increase training data. You can see that in code lines 74 - 83.

After the collection process, I had 45021 number of images (data points). I then preprocessed this data by applying Gray scale, cropping and image resizing to 64x64x3. ropping and resizing.

![alt text][image3]

Here are images after applying gray scale.

![alt text][image4]

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by my driving video. I used an adam optimizer so that manually training the learning rate wasn't necessary.

###4. Improving model

I had many issues in final part of project: my model started to work and I just needed to fine tune (add more data and more augmentation of images) but due unknown reason everything fell apart and model stopped working. Even I had backup I could not return to working model. I could not debug and find a bug. I have re-driven and add new images, I have checked, double cheeked the code and did not find what happened. Finally I rewritten the whole code and succeeded to get back on track. Due all of this I am late with my submission. 

To improve model I would add more training data, fine tune augmentation of images, definitely try to improve model with changing model layers.
