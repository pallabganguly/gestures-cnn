import cv2                 
import numpy as np         
import os        

from random import shuffle 
from tqdm import tqdm

import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

# TRAIN_DIR = '/home/pallab/gestures-cnn/images/manual'
TRAIN_DIR = '/home/pallab/Desktop/orig'

IMG_SIZE = 64 # Resizing and reshaping 
LR = 1e-3 # Learning Rate is 0.001

MODEL_NAME = 'gesture-{}-{}-{}.model'.format(LR, IMG_SIZE, '5-layer-3-conv-3FC')


def label_img(img):
    word_label = img[0]
    # conversion to one-hot array [index,v-shape,fist,terminal]
   
    if word_label == 'i': return [1,0,0,0,0]                            
    elif word_label == 'v': return [0,1,0,0,0]
    elif word_label == 'f': return [0,0,1,0,0]
    elif word_label == 't': return [0,0,0,1,0]
    elif word_label == 'b': return [0,0,0,0,1]

def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR,img)
        img = cv2.imread(path , cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data

try:
    train_data = np.load('train_data.npy')
except (FileNotFoundError):
    train_data = create_train_data()

'''
CNN Architecture:
Convolutional Layer #1: Applies 32 5x5 filters (extracting 5x5-pixel subregions), with ReLU activation function
Pooling Layer #1: Performs max pooling with a 2x2 filter and stride of 2 (which specifies that pooled regions do not overlap)

Convolutional Layer #2: Applies 64 5x5 filters, with ReLU activation function
Pooling Layer #2: Again, performs max pooling with a 2x2 filter and stride of 2

Dense Layer #1: 1,024 neurons, with dropout regularization rate of 0.5 (probability of 0.5 that any given element will be dropped during training)
Dense Layer #2 (Logits Layer): 10 neurons, one for each digit target class (0–9).
'''

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

model = tflearn.DNN(convnet, tensorboard_dir='log')

train=[]
test=[]
train = train_data[:2500]
test = train_data[2500:]

X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y = [i1[1] for i1 in train]

test_x = np.array([i2[0] for i2 in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
test_y = [i3[1] for i3 in test]

model.fit(
    {'input': X}, 
    {'targets': Y}, 
    n_epoch=10, 
    validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=50, 
    show_metric=True, 
    run_id=MODEL_NAME)

model.save("/home/pallab/gestures-cnn/tfmodels/"+MODEL_NAME)