Detects lane lines on the road--simulated using a video.

This is done by first detecting lane lines on a single frame (an image) through filtering and establishing region of interest. For more on how it happens--the code in the master branch is useful to see how the process looks like, the main difference is that the indicators for the lines detected are masked onto the road's image and it is wrapped by a loop to do that for every frame of the video.
