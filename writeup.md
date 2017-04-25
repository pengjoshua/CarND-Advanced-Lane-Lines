## Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/chesscalibration.png "Calibration"
[image2]: ./output_images/chess.png "Chess"
[image3]: ./output_images/undist.png "Undistorted"
[image4]: ./output_images/colorthreshold.png "Color Threshold"
[image5]: ./output_images/perspectivetransform.png "Perspective Transform"
[image6]: ./output_images/perspectiveplus.png "Perspective Plus"
[image7]: ./output_images/binarythreshold.png "Binary Threshold"
[image8]: ./output_images/histogram.png "Histogram"
[image9]: ./output_images/lanedetection.png "Lane Detection"
[image10]: ./output_images/pipeline.png "Pipeline"
[image11]: ./output_images/project_video_output.gif "Output"
[image12]: ./output_images/challenge_video_output.gif "Challenge"

### Camera Calibration

#### 1. Compute the camera matrix and distortion coefficients

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]


#### 2. Distortion-corrected image

I use cv2.undistort and the camera calibration matrix generated from the previous block to remove distortion in a sample chessboard calibration image and test image.

![alt text][image2]
![alt text][image3]

### Color and Binary Thresholding

#### 1. Sobel gradient thresholding in X and Y direction
I used Sobel gradient thresholding to detect changes in both horizontal and vertical directions

#### 2. HLS - Saturation channel thresholding
After the converting the RGB image to the HLS color space, I threshold the image on the saturation channel using the magnitude gradient to detect white and yellow pixels.

#### 3. Create white and yellow masks
In the image below, the green pixels are thresholded by the yellow mask and the red pixels are thresholded by the white mask.

![alt text][image4]

### Perspective Transform
Distortion can change the apparent size of an object in an image.
Distortion can change the apparent shape of an object in an image.
Distortion can cause an object's appearance to change depending on where it is in the field of view.
Distortion can make objects appear closer or farther away than they actually are.

A perspective transform corrects for image distortion by mapping the points in a given image to a new set of image points with a new perspective. I use the perspective transform to warp the image as if seeing the lane from above (bird's eye view). The perspective transform allows me to detect lane lines and compute the left and right lane curvatures.

The cv2.getPerspectiveTransform function requires 4 source and destination points. Since the lanes are approximated as trapezoids, I can use the 4 corners of a lane as the source points for a perspective transform. For this sample image, I have selected the following source and destination points.

|    Source     |  Destination  |
|:-------------:|:-------------:|
| (253, 697)    | (303, 697)    |
| (585, 456)    | (303, 0)      |
| (700, 456)    | (1011, 0)     |
| (1061, 697)   | (1011, 697)   |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image5]

### Perspective Transform pipeline method 
I create a perspective transform method for a pipeline to detect lane lines. 

* Apply thresholding and edge detection on the image
* Mask the upper 60% of pixels to remove any image artifacts such as sky, trees, clouds
* Use Hough transforms to detect the left and right lane lines
* Find the apex where the two lines intersect. 
* Form a trapezoid: select a point based on image dimensions slightly below the apex for the upper 2 corners, continue down both left and right lane lines for the bottom 2 corners
* Pass the trapezoid points (source points) along with a hardcoded set of destination points to cv2.getPerspectiveTransform to compute the perspective transformation matrix

In the image below, I use + symbols to indicate the corners of the lane trapezoid in the original and warped images.

![alt text][image6]

### Edge Detection
In addition to the Sobel first derivative operation I tried the Laplacian second derivative operation (http://docs.opencv.org/3.1.0/d5/d0f/tutorial_py_gradients.html). I used the Laplacian filter (cv2.Laplacian) and thresholded pixels to highlight only negative values (following a dark-light-dark edge fashion). The Laplacian filter generated better results than combinations of Sobel gradients.

I used these thresholding operations to create several masks to detect edges in the images. The first mask is the binary thresholded mask generated from the Laplacian filter. I apply thresholding only on the S (Saturation) channel of the image. If less than 1% of pixels were detected, then I apply the Laplacian threshold on the grayscaled image.
The second mask is a thresholding mask on the S channel pixels. The third mask is a brightness mask to reject dark lines. I combine either the second and third masks or the first and third masks. The result is a binary warped image.

![alt text][image7]

### Lane Line Detection 
Lane line detection is completed using the sliding windows technique explained in the course. I create a histogram to add up the pixel values along each column in the binary warped image. The 2 most prominent peaks in this histogram should indicate the x-position of the lane lines starting at the bottom of the image. Using this approximation, I find all the nonzero values inside a window (1/9 of image height) with a margin of 100px, and store those points as points for the lane line. I then slide the window up, and repeat the process to find nonzero values. I use the last known line location to restrict my search for new lane pixels. When the sliding window reaches the top of the image, I fit a second order polynomial to the points that were detected. This polynomial can be used to calculate the lane line values at any point.

![alt text][image8]
![alt text][image9]

### Compute Left and Right Radius of Curvature and Vehicle Position
I scale the lane pixel values to meters using the following scaling factors according to the 3.7m average lane line widths on US roads.
* Source: https://en.wikipedia.org/wiki/Lane#Lane_width
* ym_per_pix = 30/720 # meters per pixel in y dimension
* xm_per_pix = 3.7/700 # meteres per pixel in x dimension

I use these values to compute the polynomial coefficients in meters and use these coefficients to compute the left and right radius of curvature. I also compute the vehicle position assuming the camera is positioned at the center of the vehicle and and the distance between the left and right lane line midpoint and image center.

#### 1. Using a histogram
I use the scipy.signal find_peaks_cwt function to identify the peaks in the histogram. I filter the identified peaks to reject any peaks close to image edges and any peaks below a specified minimum. I apply the sliding window method on the peak histogram points to extract the lane pixels. First, I split the image into 10 windows (1/10 of image height) and starting with the bottom window, search for lane lines within the windows. I use a histogram to store the x and y coordinates of all nonzero pixels. The column with the most nonzero pixels is used to estimate the lane line. Furthermore, I remove any x or y coordinates that exceed 2 standard deviations from the mean. I use the filtered pixels and a weighted average of previous lane pixels with np.polyfit to compute a second order polynomial to fit the lane line points. 

#### 2. Using an image mask
I use the previously generated second order polynomial to create an image mask over a specific region of interest where I am most likely to find lane lines. I use this mask to remove non-lane pixels and all lane (nonzero) pixels in the region of interest to compute another second order polynomial. 

#### 3. Highlight left and right lane lines
I use a lane line finding method similar to the one used in the first lane line finding project of this course. The left lane is highlighted in red and the right lane is highlighted in blue.

### Image Processing Pipeline 
As a summary, the steps in my pipeline are:
* Color Transforms and Gradients 
* Perspective Transform
* Edge Detection
* Binary Thresholding
* Lane Line Detection

The following are processed images put through the pipeline.

![alt text][image10]

### Results and Discussion 

This pipeline was overall quite successful in identifying lane lines in the project videos. The pipeline has some difficulty keeping track of the lanes in both challenge videos. Short clips of the resulting output videos are presented below.

The harder challenge video provides no separating median strip blocking the view of opposing traffic, is filmed under different lighting conditions (or different time of day), contains sharper turns, contains a smaller lane, and contains closer distracting artifacts that take up more space in the image and obscure the lane (trees, tree shadows, foliage, grass, etc). My pipeline had difficulty with the illumination and color shade variation in the challenge videos.

This project highlighted the significance of the Convolutional Neural Network and Deep Learning based approach used in the previous project for behavioral cloning.

![alt text][image11]
![alt text][image12]