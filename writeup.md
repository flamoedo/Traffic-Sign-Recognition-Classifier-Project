# **Traffic Sign Recognition** 

## Writeup

---

**Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/feature1.png "image 2"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/feature2.png "Traffic Sign 1"
[image5]: ./examples/feature3.png "Traffic Sign 2"
[image6]: ./examples/feature4.png "Traffic Sign 3"
[image7]: ./examples/feature5.png "Traffic Sign 4"
[image8]: ./examples/feature6.png "Traffic Sign 5"
[chart1]: ./examples/histogram.png "Histogram"
[image9]: ./examples/softmax.png "Softmax Predictions"
[feature1]: ./examples/featuremap1.png "Feature Map 1"
[feature2]: ./examples/featuremap2.png "Feature Map 2"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

Here is a link to my [project code](https://github.com/flamoedo/Traffic-Sign-Recognition-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing that, although the classes are unbalanced, they follows the same distribution on each dataset.

![alt text][chart1]

### Design and Test a Model Architecture

On the preprocess the images are normalized, so the values of the array will be in a range from 0 to 1, that is most apropriated 
for the CNN learning process.
There are is no significative changes to the image after this conversion.
After that, the database is shuffled, so every time, the learning process is run, the order of images is direfent from the previous.

Here is an example of a traffic sign image after normalization.

![alt text][image2]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 1x1     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling    	| 2x2 stride, outputs 14x14x6 	|
| Convolution 1x1     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten 	    |      									|
| RELU					|												|
| Fully connected		| output 84       									|
| RELU					|												|
| Fully connected		| output 43       									|
| Softmax				|        									|

 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used adam adaptive learning rate, learning rate 0.003, 20 epochs, batch size 90.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

To achieve the validation set accuracy above 0.933, I tried many combinations of learning rate, epocks and batches size.
The validation is calculated on the end of every epock, and the test was calculated after the end of training process.

My final model results were:
* validation set accuracy of 0.933
* test set accuracy of 0.911

If a well known architecture was chosen:
* What architecture was chosen?
The LeNet achitecture was chosen. 

* Why did you believe it would be relevant to the traffic sign application?

The sizes of it inputs matches the size of the features, so it was suposed to geave good 
results on this project.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 
On the final tests, with images taken from the internet, the model was proven to be accurate.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are a sample of five German traffic signs that I found on the web:
The accuracy was calculated on 21 images.

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The fourth image might be difficult to classify because is very dark.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

One important thing to notice is that, on the softmax calculation, the prediction was grouped with signs of near context, showing 
that the model was able to distinguish among the context of the signs.

Calculated accuracy on 21 sample images: 95.24%

Here are the results of 5 prediction on Softmax:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed Limit (70km/h)      		| Speed Limit (70km/h)  									| 
| Speed Limit (100km/h)     			| Speed Limit (120km/h) 										|
| Keep right					| Keep right											|
| Keep right ahead      		| Keep right ahead					 				|
| Double Curve			| Double Curve      							|


Testing the model on 21 images, the model was able to correctly guess 20 out of 21 traffic signs, which gives an accuracy of 95.24%. This compares favorably to the accuracy on the test set of 91.1%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located among the ending cells of my Ipython notebook.

| Image	1		        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed Limit (70km/h)      		| 100%  									| 
| Speed Limit (30km/h)     			| 0% 										|
| Speed Limit (20km/h) 					| 0%											|
| Speed Limit (80km/h)      		| 0%					 				|
| Speed Limit (50km/h) 		| 0%      							|


| Image		2	        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed Limit (120km/h)      		| 95%  									| 
| Speed Limit (100km/h)     			| 5% 										|
| No passing for vehicles over 3.5 metric tons					| 0%											|
| Speed Limit (80km/h)       		| 0%					 				|
| Priority road			| 0%      							|

| Image		3	        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Keep right      		| 100%  									| 
| Go straight or right     			| 0% 										|
| Children crossing					| 0%											|
| Turn left ahead       		| 0%					 				|
| General caution			| 0%      							|

| Image		4	        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Turn right ahead      		| 100%  									| 
| Turn left ahead     			| 0% 										|
| Traffic signals					| 0%											|
| Go strainght or right       		| 0%					 				|
| Road work			| 0%      							|

| Image		5	        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Double curve      		| 100%  									| 
| Pedestrians     			| 0% 										|
| Right-of-way at the next inersection					| 0%											|
| Roundabout mandatory       		| 0%					 				|
| Wild animals crossing			| 0%      							|


For the second image, the model is relatively sure that this is Speed Limt (120 km/h), and the image does contain a Speed Limt (100 km/h). The rest of the images ware correctely predicted, on a very shure accuracy of almost 100%. So I think that the images on the test data set ware the same as the training set.

![alt text][image9]


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
This cells showns the activation of the first two convolution layers of the network.
On the first layer the images still recognizable, on the second convolution, the image begins to seen indistingishable.
On the next layers the activation became just a handfull of dots on the matrix.

![alt text][feature1]

![alt text][feature2]

