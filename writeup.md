## Writeup
---
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

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

[image1]: ./output_images/undistort_output.png "Undistorted"
[image2]: ./output_images/undistort_road.jpg "Road Transformed"
[image3]: ./output_images/binary_combo_example.jpg "Binary Example"
[image4]: ./output_images/warped_straight_lines.jpg "Warp Example"
[image5]: ./output_images/color_fit_lines.jpg "Fit Visual"
[image6]: ./output_images/example_output.jpg "Output"
[video1]: ./output_images/project_video.mp4 "Video"
## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README
### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "/code.ipynb" . 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]
I got the distortion_corrected image by using the `cv2.undistort(image,mtx,dist,None,mtx)`function. The parameter of mtx and dist have got in the step of Camera Calibration.
#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I create a threshold binary image by using the following method.

* both the x and y gradients meet the threshold criteria.
 1.Gray the image using`cv2.cvtColor()`, 
 2.Calculate the absolute of gradient in x and y direction`np.absolute(cv2.Sobel())` 
 3.Establish an zero image which has the same shape of gray`np.zeros_like()`. 
 4.Put the pixels meet the threshold criterion in zero image to one
  
* gradient magnitude and direction are both within their threshold values.
1.Calculate the gradient magnitude using the square root of the gradient in x square and y square`np.sqrt()`
2.Establish an zero image which has the same shape of gray`np.zeros_like()`. 
3.Put the pixels meet the threshold criterion in zero image to one
4.Calculate the direction of gradient`np.arctan2()`
5.Establish an zero image which has the same shape of gray`np.zeros_like()`. 
6.Put the pixels meet the threshold criterion in zero image to one

* Extract the yellow lane line  using S channal of HLS color space 
1.Convert image to HLS color space`cv2.cvtColor()`
2.Establish an zero image which has the same shape of S channel`np.zeros_like()`. 
3.Put the pixels meet the threshold criterion in S channel to one

*Extract the white lane line  using L channal of LUV color space 
1.Convert image to HLS color space`cv2.cvtColor()`
2.Establish an zero image which has the same shape of S channel`np.zeros_like()`. 
3.Put the pixels meet the threshold criterion in S channel to one
  
* Combined several selections to one part.
1.Establish an zero image which has the same shape of gray`np.zeros_like()`. 
2.`combined[(hls_s_binary==1) | (luv_l_binary==1)|((gradx==1) & (mag_binary==1) &(grady==1) & (dir_binary==1))]=1`  

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()`,.  The `warp()` function takes an image (`img`) as input. I chose the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that the perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

1. Created my histogram of the bottom half of the image`np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)`

2. Split the histogram for two lines (left and right) using a `midpoint`.

3. Set up windows and window hyperparameters`nwindows`,`margin`,`minpix`,`window_height`

4. Iterate through `nwindows` to track curvature.

5. Fit a polynomial

7. Use the previous polynomial to skip the sliding window,Than ,I can do a highly targeted search for the next frame.


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `code.ipynb` in the function `calculate_curv_and_pos()`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `code.ipynb` in the function `draw_area()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
**Problems:**
It sames to work well when I implement it in a single image,but when I put it on the video. It appeared many issues. such as :

1. The text and the lane line area is no dynamic,even the video is just a picture because of using the wrong object. 
2. When combine the whole process, it appears some variables error.etc.

**Where will your pipeline likely fail?**

1.When the shadow or sunshine appears alternately,the fitting lane line is easily to shake. 

2.When the lane line is not clear and there is a turn, it worked not good.

**What could you do to make it more robust?**
Optimized the value of the threshold and the method when extracting the lane line.
