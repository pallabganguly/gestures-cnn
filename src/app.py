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
import time
import subprocess
prefix = 'home/pallab/gestures-cnn/'
hand_cascade = cv2.CascadeClassifier(prefix + 'classifiers/' + 'cascade_30_15.xml')

TEST_DIR = "/home/pallab/Desktop/test"
IMG_SIZE = 64
LR = 1e-3
MODEL_NAME = 'gesture-{}-{}-{}.model'.format(LR, IMG_SIZE, '5-layer-3-conv-3FC')

def skinMask(frame, x0, y0, width, height ):
    global guessGesture, visualize, mod, lastgesture, saveImg
    # HSV values
    skinkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    low_range = np.array([0, 50, 80])
    upper_range = np.array([30, 200, 255])
    
    cv2.rectangle(frame, (x0,y0),(x0+width,y0+height),(0,255,0),1)
    roi = frame[y0:y0+height, x0:x0+width]
    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    #Apply skin color range
    mask = cv2.inRange(hsv, low_range, upper_range)
    
    mask = cv2.erode(mask, skinkernel, iterations = 2)
    mask = cv2.dilate(mask, skinkernel, iterations = 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, skinkernel)
    
    #blur
    mask = cv2.GaussianBlur(mask, (15,15),1)
    #cv2.imshow("Blur", mask)
    
    #bitwise and mask original frame
    res = cv2.bitwise_and(roi, roi, mask = mask)
    # color to grayscale
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    
    
    
    
    return res

def soft(num):
    pass
    # if num==1:
    #     count=2
    #     while count>0:
    #             #p = subprocess.Popen(["C:/Program Files (x86)/VideoLAN/VLC/vlc.exe","file:///C:/Users/SAURADIP/Desktop/projvid/peace.mp4"])
    #             count=count-1
    #             print(count)
    # elif num==2:
    #     print(2)
    # elif num==3:
    #     print(3)
    # elif num==4:
    #     print(4)



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
# while True:
#     ret, frame = cap.read()
#     #cv2.rectangle(frame, (300,300), (200,200), (0,255,0),0)
#     gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     hands = hand_cascade.detectMultiScale(gray, 1.7, 30, minSize=(100, 100),maxSize=(150, 150))# minSize=(50, 50),maxSize=(80, 80))
#     for (x,y,w,h) in hands:
#         #print(x)
#         #print(y)
#         #crop_img = frame[x:x+h, y:y+w]
#         cv2.rectangle(frame,(x,y),(x+w,y+h),(200,30,200),2)
#         # value = (33, 33)
#         # hsv = cv2.cvtColor(crop_img,cv2.COLOR_BGR2GRAY)
#         # blur = cv2.GaussianBlur(hsv, (5, 5), 0)
#         # thres = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
#         # ret, res = cv2.threshold(thres, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#         # # mask = cv2.inRange(hsv, np.array([0, 5, 3]), np.array([60, 255, 255]))
#         # # gaussian = cv2.GaussianBlur(mask, (11,11), 0)
#         # erosion = cv2.erode(res, None, iterations = 1)
#         # dilated = cv2.dilate(erosion,None,iterations = 1)
#         # median = cv2.medianBlur(dilated, 7)
#         # median = cv2.resize(median, (IMG_SIZE, IMG_SIZE))
#         # data = median.reshape(IMG_SIZE,IMG_SIZE,1)
#         # model_out = (model.predict([data])[0])
#         # probabilities = model_out
#         # prediction = (model_out == model_out.max(axis=0, keepdims=True)).astype(int)
#         # label = ""
#         # if np.array_equal((prediction),np.array([1.,0.,0.,0.,0.])):
#         #     label = "INDX"
#         # elif np.array_equal((prediction),np.array([0.,1.,0.,0.,0.]) ): 
#         #     label = "VSHP"
#         # elif np.array_equal((prediction) , np.array([0.,0.,1.,0.,0.])): 
#         #     label = "FIST"
#         # elif np.array_equal((prediction) , np.array([0.,0.,0.,1.,0.])): 
#         #     label = "THMB"
#         # elif np.array_equal((prediction) , np.array([0.,0.,0.,0.,1.])): 
#         #     label = "NOGS"

        
#         # gests = ["INDX", "VSHP", "FIST", "THMB", "NOGS"]
#         # plt.barh(gests, probabilities)
#         # plt.savefig("foobar.jpg")
#         # plot = cv2.imread("foobar.jpg")
#         # cv2.putText(frame, label, (200,200), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
#         cv2.imshow('cropped', frame)
#         # cv2.imshow('mask', median)
#         # cv2.imshow("Scores", plot)
#         # os.remove("foobar.jpg")
#         # # write_img = cv2.resize(median, (50,50))
#         # # cv2.imwrite('images_data/peace/'+str(count)+'.jpg',write_img)
#         # # print count
#         # # count += 1
#         k = cv2.waitKey(1) & 0xFF
#         if k == 27:
#             break
kernel = np.ones((15,15),np.uint8)
kernel2 = np.ones((1,1),np.uint8)
skinkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
low_range = np.array([0, 50, 80])
upper_range = np.array([30, 200, 255])
while count != 1001:
    plt.clf()
    ret, frame = cap.read()
    ret = cap.set(3,640)
    ret = cap.set(4,480)
    #cv2.rectangle(frame, (400,400), (200,200), (0,255,0),0)
    crop_img = frame[200:400, 200:400]
    value = (33, 33)
    roi=skinMask(frame,400,200,100,100)
    cv2.imshow('win',roi)
    # #hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # #cv2.imshow('hsv',hsv)
    # # #Apply skin color range
    # #mask = cv2.inRange(roi, low_range, upper_range)

    # mask = cv2.erode(roi, skinkernel, iterations = 1)
    # mask = cv2.dilate(mask, skinkernel, iterations = 1)

    # #blur
    # mask = cv2.GaussianBlur(mask, (15,15), 1)
    # #cv2.imshow("Blur", mask)

    # #bitwise and mask original frame
    # res = cv2.bitwise_and(frame, frame, mask = mask)
    # # color to grayscale
    # #res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    # #hands = hand_cascade.detectMultiScale(hsv, 1.7, 30, minSize=(100, 100),maxSize=(150, 150))# minSize=(50, 50),maxSize=(80, 80))
    # #hsv = cv2.cvtColor(crop_img,cv2.COLOR_BGR2GRAY)
    # #for (x,y,w,h) in hands:
    #     #cv2.rectangle(frame,(x,y),(x+w,y+h),(200,30,200),2)
    # gray=roi
    blur = cv2.GaussianBlur(roi, (5, 5), 0)
    thres = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,3)
    ret, res = cv2.threshold(thres, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # mask = cv2.inRange(hsv, np.array([0, 5, 3]), np.array([60, 255, 255]))
    # gaussian = cv2.GaussianBlur(mask, (11,11), 0)
    erosion = cv2.erode(res, None, iterations = 2)
    dilated = cv2.dilate(erosion,None,iterations = 1)
    median = cv2.medianBlur(dilated, 7)
    median = cv2.resize(median, (IMG_SIZE, IMG_SIZE))
    data = median.reshape(IMG_SIZE,IMG_SIZE,1)
    model_out = (model.predict([data])[0])
    probabilities = model_out
    prediction = (model_out == model_out.max(axis=0, keepdims=True)).astype(int)
    label = ""
    if np.array_equal((prediction),np.array([1.,0.,0.,0.,0.])):
        label = "INDEX"
        #time.sleep(1)
        soft(1) # invoke software 
    elif np.array_equal((prediction),np.array([0.,1.,0.,0.,0.]) ): 
        label = "V-SHAPE"
        #time.sleep(1)
        soft(2)
    elif np.array_equal((prediction) , np.array([0.,0.,1.,0.,0.])): 
        label = "FIST"
        #time.sleep(1)
        soft(3)
    elif np.array_equal((prediction) , np.array([0.,0.,0.,1.,0.])): 
        label = "THUMB"
        #time.sleep(1)
        soft(4)
    elif np.array_equal((prediction) , np.array([0.,0.,0.,0.,1.])): 
        label = "NO-GESTURE"
    


    #print(probabilities)
    gests = ["INDEX ("+str(round(probabilities[0]*100,2))+" % )", "V-SHAPE ("+str(round(probabilities[1]*100,2))+" % )", "FIST ("+str(round(probabilities[2]*100,2))+" % )", "THUMB ("+str(round(probabilities[3]*100,2))+" % )", "NO-GES ("+str(round(probabilities[4]*100,2))+" % )"]
    lists=plt.barh(gests, (np.around(probabilities,decimals=1)))
    if  np.around(probabilities,decimals=1).any() > 0.7:
        lists[probabilities.argmax()].set_color('g')

   # plt.set_xticklabels(np.around(np.multiply(probabilities,100),decimals=2), minor=False)
    plt.xticks(np.arange(0, 1, step=0.2))
    # plt.savefig("foobar.jpg")
    plt.title('Gesture Recognition')
    plt.xlabel('Probabilities of Detection')
    plt.ylabel('Gestures')
    plt.figure(figsize=(14,7))
    # plot = cv2.imread("foobar.jpg")
    cv2.putText(frame, label, (400,200), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow('cropped', frame)
    cv2.imshow('mask', median)
    # cv2.imshow("Scores", plot)
    # os.remove("foobar.jpg")
    # write_img = cv2.resize(median, (50,50))
    # cv2.imwrite('images_data/peace/'+str(count)+'.jpg',write_img)
    # print(count)
    count += 1
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
