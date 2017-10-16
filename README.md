# **Vehicle Detection** 

## Fifth and final project for Udacity's Self Driving Car Nanodegree - Term 1

### This is the project writeup. [Instructions](Instructions.md) are here. Project's expectations are called rubic points and they are [here](https://review.udacity.com/#!/rubrics/513/view).

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image_car]: ./examples/car.png
[image_noncar]: ./examples/non_car.png
[image_noncar]: ./examples/hog.png
[image_car-box]: ./examples/car-box.png
[image_car-box2]: ./examples/car-box2.png
[image_car-box3]: ./examples/car-box3.png
[image_heatmap]: ./examples/heatmap.png
[image_heatmap2]: ./examples/heatmap2.png

[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the second to fifth of the IPython notebook.

In the second cell, I am reading all the `car` and `non-car` images. In the third cell, I am showing example image of each category. 

![alt text][image_car] ![alt text][image_noncar]

Following two cells have few lines of code from project module and rest of the HOG logic.

![alt text][image_hog]

I then extracted features and explored different color spaces / parameters in code cell six and seven. In particular, I tried HLS and YCrCb because many other students had success and recommended these on the forums. I reused project module quizzes functions.


```
# Define parameters for feature extraction
color_space = 'HLS' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 6 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
```


#### 2. Explain how you settled on your final choice of HOG parameters.

I started with YCrCb color space until lately. But, I came to realization that HLS outperformed YCrCb in the final video. So, I decided to settle on HLS.

I also played with other parameters like orient, pix_per_cell, spatial_size, and hist_bins. I found that these helped determine how quickly the video was processed and settled on the following parameters. 

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using LinearSVC as recommended in the lessons. It took 8.04 seconds to train SVC and I achieved 98.79% accuracy. All runs had similar results.

I used total three feature vectors. One is spatial binning to get the raw color info, second is using a histogram of the color spectrum to get color info, and third is the HOG features to get the shape info. I concatenate all the features to give the feature vector.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I performed a sliding window search in code cell 8. In my final pipeline, I utilized the following parameters:

```
    xy_window = [[80, 80], [96, 96], [128, 128]]
    x_start_stop = [[200, None], [200, None], [412, 1280]]
    y_start_stop = [[390, 540], [400, 600], [400, 640]]
```

I chose the x_start_stop parameters because there were no cars to the left that needed detecting. In addition, there were times when the video detected false positives on the lighter colored pavement or in the shadows of the trees.

y_start_stop was chosen due to the xy_window parameters.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using HLS 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image_car-box] ![alt text][image_car-box2] ![alt text][image_car-box3]


---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected. I implemented a bounding box averaging system using deque.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

Some heat map visuals, rest are in IPython notebook.

![alt text][image_heatmap] ![alt text][image_heatmap2] 

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The size of the bounding box does not flow smoothly between frames. 

When two cars are overtaking, it treats both as the same car. track individual positions more accurately over time.

The data processing pipeline is extremely slow. 

