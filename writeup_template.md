#**Udacity Self Driving Car - Behavioral Cloning** 

##Ante Sladoejvic - Writeup

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 16 and 64 (model.py lines 138-157) 

The model includes RELU layers to introduce nonlinearity (code lines 138-157), and the data is normalized in the model using a within python function read_image_gray_norma (code line 29). 


####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 149,151,153). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 157).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and made around 6 laps in one direction and 3 laps in oposite direction

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to 

My first step was to use a convolution neural network model similar to the Nvidia research paper I thought this model might be appropriate because it worked.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that Nvidia model had a low mean squared error and did not succeed to drive whole lap. On the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model by copying Nvidia model, adding dropout and MaxPooling layers.

Then I flattened model from 1000 -> 500 -> 100 -> 20 -> 1

The final step was to run the simulator to see how well the car was driving around track one. 
While i was improving my model vehicle was constantly felling off the track. I was using each drive to improve model. From fine tuning model to adding cropping, resizing image and graying out images.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 138-157) consisted of a convolution neural network with the following layers and layer sizes:

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

Here is summary of how network looks:
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
convolution2d_1 (Convolution2D)  (None, 62, 62, 16)    448         convolution2d_input_1[0][0]      
(16, 3, 3)
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 31, 31, 16)    0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 29, 29, 32)    4640        maxpooling2d_1[0][0]             
(32, 3, 3)
____________________________________________________________________________________________________
maxpooling2d_2 (MaxPooling2D)    (None, 14, 14, 32)    0           convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 12, 12, 32)    9248        maxpooling2d_2[0][0]             
(32, 3, 3)
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 10, 10, 64)    18496       convolution2d_3[0][0]            
(64, 3, 3)
____________________________________________________________________________________________________
maxpooling2d_3 (MaxPooling2D)    (None, 5, 5, 64)      0           convolution2d_4[0][0]            
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 3, 3, 64)      36928       maxpooling2d_3[0][0]             
(64, 3, 3)
____________________________________________________________________________________________________
maxpooling2d_4 (MaxPooling2D)    (None, 1, 1, 64)      0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 64)            0           maxpooling2d_4[0][0]             
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 1000)          65000       flatten_1[0][0]                  
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 1000)          0           dense_1[0][0]                    
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 500)           500500      dropout_1[0][0]                  
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 500)           0           dense_2[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 100)           50100       dropout_2[0][0]                  
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 100)           0           dense_3[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 20)            2020        dropout_3[0][0]                  
____________________________________________________________________________________________________
dense_5 (Dense)                  (None, 1)             21          dense_4[0][0]                    
====================================================================================================
Total params: 687,401
Trainable params: 687,401
Non-trainable params: 0
____________________________________________________________________________________________________
Epoch 1/5

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded 6 laps on track one using center lane driving. Here is an example image of center lane driving after cropping:

![alt text][image2]

Here are images after applying gray scale, cropping and resizing

![alt text][image3]
![alt text][image4]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to get back on track. 

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
