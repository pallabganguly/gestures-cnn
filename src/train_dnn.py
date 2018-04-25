import cv2                 
import numpy as np         
import os        

from random import shuffle 
from tqdm import tqdm      
# Contains 3000 Non Varying Image

# TRAIN_DIR = '/home/sauradip/Desktop/cnn-tf/data/raw-data/mix'
# TEST_DIR = '/home/sauradip/Desktop/cnn-tf/data/raw-data/mix'

# Contains 5000+ Variation IMages

TRAIN_DIR = 'C:/Users/SAURADIP/Desktop/cnn/resized'
TEST_DIR = 'C:/Users/SAURADIP/Desktop/cnn/resized'

IMG_SIZE = 100 # Resizing and reshaping 
LR = 1e-3 # Learning Rate is 0.001
MODEL_NAME = 'gesture-{}-{}.model'.format(LR, '6_CNN-basic') # just so we remember which saved model is which, sizes must match

def label_img(img):
    word_label = img[0]
    # conversion to one-hot array [index,v-shape,fist,thumb]
   
    if word_label == 'i': return [1,0,0,0]                            
    elif word_label == 'v': return [0,1,0,0]
    elif word_label == 'f': return [0,0,1,0]
    elif word_label == 't': return [0,0,0,1]

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

#train_data = create_train_data()
# If i have already created the dataset ,i will just load it :
try:
    test_data = np.load('test_data.npy')
except (FileNotFoundError):
    test_data = process_test_data()
    

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

import tensorflow as tf

train=[]
test=[]
train = train_data[1000:4000]
test = train_data[-1000:]

X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y = [i1[1] for i1 in train]

test_x = np.array([i2[0] for i2 in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
test_y = [i3[1] for i3 in test]

with tf.device('/gpu:0'):
    tf.reset_default_graph()
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
	
    #tflearn.init_graph(num_cores=4,gpu_memory_fraction=0.5)
    convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
    #print (sess.run(convnet))
    # 7 Layered CNN Architecture with 5 Convolutional and 2 FC Layer with Dropout and Max Pooling
    #Layer 1
    convnet = conv_2d(convnet, 32, 5, activation='relu')
    #convnet = max_pool_2d(convnet, 2)
    convnet = tflearn.layers.normalization.batch_normalization (convnet, beta=0.0, gamma=1.0, epsilon=1e-05, decay=0.9, stddev=0.002,
              trainable=True, restore=True, reuse=False, scope=None, name='BatchNormalization1')
    convnet = dropout(convnet, 0.5)
    print(convnet.shape)

    #Layer 2
    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = tflearn.layers.normalization.batch_normalization (convnet, beta=0.0, gamma=1.0, epsilon=1e-05, decay=0.9, stddev=0.002,
              trainable=True, restore=True, reuse=False, scope=None, name='BatchNormalization1')
    convnet = dropout(convnet, 0.5)
    convnet = max_pool_2d(convnet, 2)
    print(convnet.shape)

    #Layer 3
    convnet = conv_2d(convnet, 128, 5, activation='relu')
    #convnet = max_pool_2d(convnet, 2)
    convnet = tflearn.layers.normalization.batch_normalization (convnet, beta=0.0, gamma=1.0, epsilon=1e-05, decay=0.9, stddev=0.002,
              trainable=True, restore=True, reuse=False, scope=None, name='BatchNormalization1')
    convnet = dropout(convnet, 0.5)
    #print(convnet.shape)

    #Layer 4
    convnet = conv_2d(convnet, 256, 5, activation='relu')
    convnet = tflearn.layers.normalization.batch_normalization (convnet, beta=0.0, gamma=1.0, epsilon=1e-05, decay=0.9, stddev=0.002,
              trainable=True, restore=True, reuse=False, scope=None, name='BatchNormalization1')
    convnet = dropout(convnet, 0.5)
    convnet = max_pool_2d(convnet, 2)

    print(convnet.shape)

    #Layer 5
    convnet = conv_2d(convnet, 256, 5, activation='relu')
    convnet = tflearn.layers.normalization.batch_normalization (convnet, beta=0.0, gamma=1.0, epsilon=1e-05, decay=0.9, stddev=0.002,
              trainable=True, restore=True, reuse=False, scope=None, name='BatchNormalization1')
    convnet = dropout(convnet, 0.5)
    convnet = max_pool_2d(convnet, 2)

    print(convnet.shape)
    
    convnet = fully_connected(convnet, 4, activation='softmax')
    print(convnet.shape)
    convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
    
    model = tflearn.DNN(convnet, tensorboard_dir='log')
    model.fit({'input': X}, {'targets': Y}, n_epoch=15, validation_set=({'input': test_x}, {'targets': test_y}), 
        snapshot_step=50, show_metric=True, run_id=MODEL_NAME)

#model.save("/home/pallab/gestures-cnn/tfmodels/"+MODEL_NAME)
model.save("C:/Users/SAURADIP/Desktop/cnn/model/"+MODEL_NAME)
