import os
import numpy as np
import cv2

os.chdir("/home/pallab/gestures-cnn/images/resized")
i = 0
for image in os.listdir("."):
    img = cv2.imread(image, 0)
    rows, cols = img.shape
    # Make some noise
    mean, sd = 0, 50**0.5
    gaussnoise = np.random.normal(mean,sd,(rows,cols))
    gaussnoise = gaussnoise.reshape(rows,cols)
    noisy = dst + gaussnoise
    cv2.imwrite("/home/pallab/gestures-cnn/images/resized/"+str(image)+"_"+str(i)+".jpg", dst)