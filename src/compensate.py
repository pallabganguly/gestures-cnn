import os
import cv2
import numpy as np

FROM = "/home/pallab/gestures-cnn/raw-data/thumb"
TO = "/home/pallab/gestures-cnn/images/resized/"
i = 0
os.chdir(FROM)
for image in os.listdir(".")[:300]:
    im = cv2.imread(image, 0)
    crop = im[200:920, 0:720]
    rows, cols = crop.shape
    blur = cv2.GaussianBlur(crop, (33, 33), 0)
    erosion = cv2.erode(crop, None, iterations = 1)
    dilated = cv2.dilate(erosion,None,iterations = 1)
    median = cv2.medianBlur(dilated, 7)

    tranx = 100*np.random.uniform() #- transrange/2
    trany = 100*np.random.uniform() #- transrange/2
    M = np.float32([[1,0,tranx],[0,1,trany]])
    dst = cv2.warpAffine(median,M,(cols,rows))

    cx = 100*np.random.uniform()
    cy = 100*np.random.uniform()
    angle = np.random.uniform(low = -10.0, high = 10.0)
    M = cv2.getRotationMatrix2D((cx/2,cy/2),angle,1)
    dst = cv2.warpAffine(dst,M,(cols,rows))

    toggle = round(np.random.uniform())
    if toggle:
        mean, sd = 0, 50**0.5
        noise = np.random.normal(mean,sd,(rows,cols))
        noise = noise.reshape(rows, cols)
        dst = dst + noise

    cv2.imwrite(TO+"t_6"+str(i)+".jpg", dst)
    i += 1

print(str(i)+" files were processed")
if i!=300:
    print("Some files were not processed")
else:
    print("Done")