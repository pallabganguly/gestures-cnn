'''
Objective: 
1. to crop the images to 30x30
2. convert to grayscale
3. Apply Gaussian Blur
4. Apply dilation
5. Apply some kind of mask? (Why?)
'''

import os
import cv2
import numpy as np

# This is the path from root directory to this project
PATH = "/home/pallab/gestures-cnn/raw-data/"
IMG_DIR = "/home/pallab/gestures-cnn/images/orig/"
# PATH = "/home/pallab/Desktop/"
print("Processing files ....")
k = 0
gestures = ["index", "fist", "thumb", "vsign"]
for gesture in gestures:
    os.chdir(PATH+gesture)
    c = 0
    for image in os.listdir("."):
        img = cv2.imread(image)
        # crop = img[120:840, 0:720]
        crop = img[150:870, 0:720]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(hsv, (5, 5), 0)
        thres = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
        ret, res = cv2.threshold(thres, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        # mask = cv2.inRange(hsv, np.array([0, 5, 3]), np.array([60, 255, 255]))
        # gaussian = cv2.GaussianBlur(mask, (11,11), 0)
        erosion = cv2.erode(res, None, iterations = 1)
        dilated = cv2.dilate(erosion,None,iterations = 1)
        median = cv2.medianBlur(dilated, 7)
        # res = cv2.resize(median, (30, 30))
        # filename = IMG_DIR + str(gesture[0])+str(c)+".jpg" # changed here
        filename = "/home/pallab/Desktop/orig/"+str(gesture[0])+str(c)+".jpg"
        # print(c)
        cv2.imwrite(filename, median)
        c += 1

    k += c

print("Processed", k, "files")
