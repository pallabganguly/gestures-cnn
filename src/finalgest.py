import cv2                 
import numpy as np         
import os       

from random import shuffle 
from tqdm import tqdm   

import matplotlib.pyplot as plt

import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

TEST_DIR = "/home/pallab/Desktop/test"
IMG_SIZE = 64
LR = 1e-3
MODEL_NAME = 'gesture-{}-{}-{}.model'.format(LR, IMG_SIZE, '5-layer-3-conv-3FC')

def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR,img)
        img_num = img[0]
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img), img_num])
        
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data

tf.reset_default_graph()
convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 2)
convnet = local_response_normalization(convnet)
print(convnet.shape)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = local_response_normalization(convnet)
convnet = max_pool_2d(convnet, 2)
print(convnet.shape)

convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 2)
convnet = local_response_normalization(convnet)
print(convnet.shape)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.5)

convnet = fully_connected(convnet, 512, activation='relu')
convnet = dropout(convnet, 0.5)

convnet = fully_connected(convnet, 5, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet)
model.load("/home/pallab/gestures-cnn/tfmodels/"+MODEL_NAME)

cap = cv2.VideoCapture(0)
count = 1
font = cv2.FONT_HERSHEY_DUPLEX
while count != 1001:
    ret, frame = cap.read()
    cv2.rectangle(frame, (300,300), (200,200), (0,255,0),0)
    crop_img = frame[200:300, 200:300]
    value = (33, 33)
    hsv = cv2.cvtColor(crop_img,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(hsv, (5, 5), 0)
    thres = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    ret, res = cv2.threshold(thres, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # mask = cv2.inRange(hsv, np.array([0, 5, 3]), np.array([60, 255, 255]))
    # gaussian = cv2.GaussianBlur(mask, (11,11), 0)
    erosion = cv2.erode(res, None, iterations = 1)
    dilated = cv2.dilate(erosion,None,iterations = 1)
    median = cv2.medianBlur(dilated, 7)
    median = cv2.resize(median, (IMG_SIZE, IMG_SIZE))
    data = median.reshape(IMG_SIZE,IMG_SIZE,1)
    model_out = (model.predict([data])[0])
    prediction = (model_out == model_out.max(axis=0, keepdims=True)).astype(int)
    label = ""
    if np.array_equal((prediction),np.array([1.,0.,0.,0.,0.])):
        label = "INDX"
    elif np.array_equal((prediction),np.array([0.,1.,0.,0.,0.]) ): 
        label = "VSHP"
    elif np.array_equal((prediction) , np.array([0.,0.,1.,0.,0.])): 
        label = "FIST"
    elif np.array_equal((prediction) , np.array([0.,0.,0.,1.,0.])): 
        label = "THMB"
    elif np.array_equal((prediction) , np.array([0.,0.,0.,0.,1.])): 
        label = "NOGS"

    cv2.putText(frame, label, (200,200), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow('cropped', frame)
    cv2.imshow('mask', median)
    # #
    # write_img = cv2.resize(median, (50,50))
    # cv2.imwrite('images_data/peace/'+str(count)+'.jpg',write_img)
    # print count
    # count += 1
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
