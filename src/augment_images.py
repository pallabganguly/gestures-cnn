import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

os.chdir("/home/pallab/gestures-cnn/images/orig")
i, j = 0, 0
transrange = 100
print("Processing images ....")
for image in os.listdir("."):
    img = cv2.imread(image, 0)
    rows, cols = img.shape

    # Translate Images
    tranx = transrange*np.random.uniform() #- transrange/2
    trany = transrange*np.random.uniform() #- transrange/2
    M = np.float32([[1,0,tranx],[0,1,trany]])
    dst = cv2.warpAffine(img,M,(cols,rows))

    # Rotate Images
    cx = transrange*np.random.uniform()
    cy = transrange*np.random.uniform()
    angle = np.random.uniform(low = -10.0, high = 10.0)
    M = cv2.getRotationMatrix2D((cx/2,cy/2),angle,1)
    dst = cv2.warpAffine(dst,M,(cols,rows))

    # Make some noise
    switch = round(np.random.uniform())
    if switch:
        mean, sd = 0, 50**0.5
        gaussnoise = np.random.normal(mean,sd,(rows,cols))
        gaussnoise = gaussnoise.reshape(rows,cols)
        img = img + gaussnoise
    # dst = cv2.resize(dst, (int(0.75*cols), int(0.75*rows)), interpolation=cv2.INTER_CUBIC)

    cv2.imwrite("/home/pallab/gestures-cnn/images/resized/"+image, img)
    cv2.imwrite("/home/pallab/gestures-cnn/images/resized/"+str(image)[:-4]+"_"+str(i)+".jpg", dst)
    i += 1

print("Done")