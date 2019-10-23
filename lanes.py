# This program takes a digital input and
# outputs a filtered version of it after
# identifying lane lines

import cv2
import numpy as np

# Presenting original image
def original(image):
    cv2.imshow('result- original', image)
    cv2.waitKey(0)

# Presenting image in gray scale
def gray(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    cv2.imshow('result- gray', gray)
    cv2.waitKey(0)

# Presenting blurred image
def blur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    cv2.imshow('result- gaussian blur 5x5', blur)
    cv2.waitKey(0)

# Returning gradient of image
def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

# Region of interest: a static area from the input image where the lanes
# will be identified in.
def ROI(image):
    height = image.shape[0]                     # Number of rows
    polygons = np.array([
    [(200, height), (1100, height), (550, 250)] # Coordinates taken from matplotlib's pyplot image show function
    ])
    mask = np.zeros_like(image)                 # mask will have same dimenions of image but all black pixels
    cv2.fillPoly(mask, polygons, 255)           # 255 means filling in white the masked pixels
    masked_image = cv2.bitwise_and(image, mask) # Applying bitwise-and to return gradient values inside ROI
    return masked_image

# Displaying the lane lines over the image
def display_lines(image, lines):
        line_image = np.zeros_like(image)       # Initializing a black image to display all the detected lines onto it
        if lines is not None:                   # Checking that lines where detected
            for line in lines:
                x1, y1, x2, y2 = line.reshape(4)    # Reshaping to array that each contains 4 variable values (unpacking the 4 values into 4 variables)
                #   Drawing the lines on the black image I just created--coordinates provided based on lines, and finally choosing color & thickness
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 13)
        return line_image

# Getting the coodinates of the averaged lines
def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3 / 5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

# For smoother detection (averaging all lines detect on each side for a singl line per side)
def average_slope_intercept(image, lines):
    left_fit = [] # Coordinates of the lines on the left
    right_fit = [] # On the right
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1) # Return a vector of coefficients that describes the slope, fitting a polynomial of degree 1
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept)) # Appending each slope as a tuple
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis = 0)
    right_fit_average = np.average(right_fit, axis = 0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])



# Running the methods to see what happens at each part of the detection process 
#image = cv2.imread('test_image.jpg')
#lane_image = np.copy(image)
#canny_image = canny(lane_image)
#cropped_image = ROI(canny_image)
#lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
#averaged_lines = average_slope_intercept(lane_image, lines)
#line_image = display_lines(lane_image, averaged_lines) # lines variable for not the averaged lines (all detected lines)
#aligned_image = cv2.addWeighted(lane_image, 0.8, line_image, 110, 0) # aligning both images to have makred lines representing detected lines
#original(lane_image)
#gray(lane_image)
#cv2.imshow('result- gradient', canny_image)
#cv2.waitKey(0)
#cv2.imshow('result- ROI', cropped_image)
#cv2.waitKey(0)
#cv2.imshow('result- line image', line_image)
#cv2.waitKey(0)
#cv2.imshow('result- aligned image', aligned_image)
#cv2.waitKey(0)

# Running the code over every frame of the video
cap = cv2.VideoCapture("test2.mp4")
while(cap.isOpened()):
    _, frame = cap.read() # Getting frames from the video
    canny_image = canny(frame)
    cropped_image = ROI(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    aligned_image = cv2.addWeighted(frame, 0.8, line_image, 100, 0)
    cv2.imshow("result", aligned_image)
    cv2.waitKey(1) # Waits 1ms in between frames to display next one
