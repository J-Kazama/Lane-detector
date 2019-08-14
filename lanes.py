# This program takes digital input and
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


# Running the methods to see what each stage provides
image = cv2.imread('test_image.jpg')
lane_image = np.copy(image)
canny = canny(lane_image)
original(lane_image)
gray(lane_image)
cv2.imshow('result- gradient', canny)
cv2.waitKey(0)
cv2.imshow('result- ROI', ROI(canny))
cv2.waitKey(0)
