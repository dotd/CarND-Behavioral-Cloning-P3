# Behavioral Cloning Project

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior - DONE!
* Build, a convolution neural network in Keras that predicts steering angles from images - DONE!
* Train and validate the model with a training and validation set - DONE!
* Test that the model successfully drives around track one without leaving the road - DONE!
* Summarize the results with a written report - DONE!

[//]: # (Image References)

[image1]: ./examples/code.png "Model Code"
[image2]: ./examples/center_2017_03_19_14_51_59_025.jpg "Center of the Lane"
[image3]: ./examples/center_2017_03_19_14_51_40_374.jpg "Recovery Image"
[image6]: ./examples/flip.png "Flipped image"

## Files Submitted & Code Quality

### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py - containing the script to create and train the model
* drive.py - for driving the car in autonomous mode
* model.h5 - containing a trained convolution neural network 
* writeup_report.md - summarizing the results
* Main.ipynb - the main code
* final_movie.mp4 - movie to demonstrate the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

## Model Architecture and Training Strategy

### 1. An appropriate model architecture has been employed

My model consists of 5 convolution neural network layers with 5x5 filter sizes and depths between 24, 36, 48, 15, and 8 (model.py lines 67-73). Between these layers 2 maxpooling standard layers are given.  

Afterward, all connected layers are introduced (output dimensions of 150, 100, 50, 1) to mildly reduce the dimension to a regression layer (lines 76-79). 

The model includes RELU layers to introduce nonlinearity and the data is normalized in the model using a Keras lambda layer (code line 66). 

#### Analysis and Process of Development
I started with a much simpler networks. I've noticed in the test phase, as well in the simulator, that the driving of the car was too rough. Therefore, I've looked for a way to fit more accurately to the behavior that I want, in the cost at first of overfitting. I based the network part of it on the nVidia network (layers 1-3) and part on the classical LetNet (the rest of the network).  

### 2. Attempts to reduce overfitting in the model

In order to cope with overfitting, I used both the left and right images that the simulator produced. Also, I generated myself the data and played in the simulator no standard behavior such as recovering from going off the road, as well providing much more than the default data provided. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. Actually, since the parameter space is quite big, I preffered not play with this part of the project. 

### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road (as mentioned above).

For details about how I created the training data, see the next section. 

## Model Architecture and Training Strategy

### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to make a solution both from nVidia architecure as well as from LeNet architecture. Also, I 

My first step was to use a convolution neural network model similar to the LeNet. I thought this model might be appropriate because we are dealing only with "pictures". Eventually, we needed more flexible network to approximate the data appropriately.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that it will includes some pooling layers. Note that no regularization nor dropout was needed, as evident from the movie.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. to improve the driving behavior in these cases, I simulated more data with the problematic sections of the road.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

### 2. Final Model Architecture

Here is the relevant code of the architecture:

![alt text][image1]

### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to get into the middle of the lane. These images show what a recovery looks like starting from:

![alt text][image3]

Then I repeated this process on track two in order to get more data points.

To augment the data set, I also flipped images and angles thinking that this would increase the model learnability. For example, here is an image that has then been flipped:

![alt text][image6]

After the collection process, I had X number of data points. I then preprocessed this data by ...

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3-4 as evidenced by the run result. I used an adam optimizer so that manually training the learning rate wasn't necessary.
