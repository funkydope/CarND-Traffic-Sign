#**Traffic Sign Recognition Write Up** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./Figures/Class_Hist.png "Sign Class Histogram"
[image2]: ./Figures/Visualize_Speed_Signs.png "Speed Signs"
[image3]: ./Figures/Learning_Rate.png "Validation Accuracy during Training"
[image4]: ./Figures/Softmax_Web-Signs.png "Softmax Web Signs"

###Data Set Summary & Exploration

####1. Data Summary

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Exploration

Here is a hisogram of the data set seperated into sign classes. There is huge variation is the number of images for each class. I would think that a more even data set would train better.

![alt text][image1]

###Design and Test a Model Architecture

####1. I preprocessed the images by converting them to grayscale first. This reduces the number of inputs to the network and significantly reduces training time. I then normalized the images to a mean of 0 and standard deviation of 1 based on the training data. This allows the weights in the network to operate effectively on their inputs.

The figure below shows the original RGB image, the grayscale image, and the normalized grayscale image. PyPlot automatically displays images relative to their input ranges. This increases the contrast of many of the darker images in the data set. This also makes the gray and normalized images to appear the same.

Correcting the brightness and contrast of many of these images likely would have improved training.

![alt text][image2]


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My model architecture is based off of VGGNet. VGGNet only uses 3x3 filters in its convolutional layers. Research has shown that small spatial filters with large depths operate more efficiently in conv nets. Research has also shown that only one fully connected layer is needed. Additional FC layers expand the number of parameters without improving model performance. Testing my network confirmed this.

 I used 2 max pooling layers in between my conv layers. This reduced the data spatially while increasing the depth.
 
 An average pooling layer placed after my conv layers reduces the output to a 1x1. This has been found to be the most efficient method of feeding the FC layer.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| 1 Convolution 3x3 filter	| 1x1 stride, same padding, outputs 32x32x16 	|
| RELU					|												|
| 2 Convolution 3x3 filter	| 1x1 stride, same padding, outputs 32x32x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x32 				|
| 3 Convolution 3x3 filter	| 1x1 stride, same padding, outputs 16x16x32 	|
| RELU					|												|
| 4 Convolution 3x3 filter	| 1x1 stride, same padding, outputs 16x16x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 8x8x64 				|
| 5 Convolution 3x3 filter	| 1x1 stride, same padding, outputs 8x8x64 	|
| RELU					|												|
| Average pooling	      	| 8x8 stride,  outputs 1x1x64 				|
| 6 Fully connected		| outputs 43 classes        									|

####3. I trained my model based on the methods used by the LeNet lab. I used the Adam Optimizer, which is the current best practice method based on Stochastic Gradient Descent. My used a 1024 sample batch size, 100 epochs, 0.01 kear

| Hyperparameter         		|     Value	        					| 
|:---------------------:|:---------------------------------------------:| 
| Batch Size         		| 100  							| 
| Epochs| 100 	|
| Learning Rate| 0.01 	|
| Decay Rate| 0.04 	|
| Weight Init SD| sqrt(2/n inputs) 	|
| Dropout | 0.5|

####4. Results


My final model results were:
* validation set accuracy of 0.941
* test set accuracy of 0.934
 
I used an iterative approach to creating my architecture.

I first used the LeNet lab since I knew the code for it worked. This achieved a ~0.90 validation accuracy and trained very quickly. I then created my own architecture based on VGGNet. I chose VGGNet because it performs very well and is easy to implement and understand. It uses only 3x3 convolutional filters. However, my model is much deeper than LeNet. This theoretically increases the complexity of image recognition of the network but also increases training time.

I initially had diffculty getting my model to train. I found that this was because of improper weight initialization. Current best practice is to use sqrt(2/n) for standard deviation of the random normalized parameters where n is the number of inputs to a given layer. This got model to train properly.

I then started playin around with constant learning rates and found the 0.1 was the highest I could go before the model wouldn't train at all.  I then applied exponential decay to reduce the learning rate during training. This allows faster training while preventing training to plateau.

![alt text][image3]

All the lines in the figure above use an intial learning rate of 0.01. However, they vary in decar rate. The blue has a decay rate of 0.4. This was the best setting and brought my validation accuracy above the criteria to 0.941.

### Test a Model on New Images

![alt text][image4]

My model correcty predicted 5 out of my 6 test images from the web. It was alsmost 100% certain for all the images, even for the incorrectly categorized one. The model likely has trouble classifying speed limit signs. It classified the 100 speed limit sign as 30. It makes sense that the model would have the most trouble here since the image details that distguish different speed signs is very small. However, it's surprising that softmax calculates such a high certainty for the 30 speed sign. I'm not sure what's going on here.

