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


"""
TEST DATA FOLDER GOES HERE, PLS REPLACE AS PER YOUR SYSTEM
"""
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

try:
    test_data = np.load('test_data.npy')
except (FileNotFoundError):
    test_data = process_test_data()

show = 50 # number of images to be shown
col = 5 # no of columns to be displayed
fig = plt.figure()
labelss = []
predics = []
shuffle(test_data)
for num,data in enumerate(test_data[:show]):
    img_num = data[1]
    img_data = data[0]
    if img_num == "i":
        labelss.append(1)
    elif img_num == "v":
        labelss.append(2)
    elif img_num == "f":
        labelss.append(3)
    elif img_num == "t":
        labelss.append(4)
    elif img_num == "b":
        labelss.append(5)
    
    str_label = ""
    y = fig.add_subplot((show/col),col,num+1)
    orig = img_data 
    data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
    # model_out = model.predict([data])[0]
    model_out = (model.predict([data])[0])
    hypths = (model_out == model_out.max(axis=0, keepdims=True)).astype(int)
    
    if np.array_equal((hypths),np.array([1.,0.,0.,0.,0.])):
        plt.title("INDX")
        predics.append(1)
    elif np.array_equal((hypths),np.array([0.,1.,0.,0.,0.]) ): 
        plt.title("VSHP")
        predics.append(2)
    elif np.array_equal((hypths) , np.array([0.,0.,1.,0.,0.])): 
        plt.title("FIST")
        predics.append(3)
    elif np.array_equal((hypths) , np.array([0.,0.,0.,1.,0.])): 
        plt.title("THMB")
        predics.append(4)
    elif np.array_equal((hypths) , np.array([0.,0.,0.,0.,1.])): 
        plt.title("NOGS")
        predics.append(5)
    
        
    y.imshow(orig,cmap='gray')
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)

plt.tight_layout()
# plt.show()

from sklearn.metrics import classification_report
print(classification_report(labelss, predics))
labelss, predics = np.array(labelss), np.array(predics)
x = 0
for v in labelss==predics:
    if v: x+=1
print("Accuracy =", x/show)