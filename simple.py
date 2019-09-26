# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 11:03:45 2019

@author: jwang
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 13:36:09 2019

@author: jwang
"""
## test 2, version 2, trained by all three "hsv" channels of the images, and the most simplifed CNN

import json 
import pandas as pd 
import matplotlib.pyplot as plt
import cv2

#%%

#load training data
"""
# as mentioned, there are 5 images in which there are no corners (totally blocked by pillar) and the ground truth 
in JSON are [[ ]]. These images are IMG_0492.JPG, IMG_0688.JPG, IMG_1343.JPG, IMG_5199.JPG, and IMG_6608.JPG. 
There are another 11 images was half blocked and thus the corresponding ground truth labels can be considered 
as errors including IMG_0493.JPG, IMG_1342.JPG, IMG_1344.JPG, IMG_5196.JPG, IMG_5197.JPG, IMG_5198.JPG, IMG_5200.JPG, 
IMG_5201.JPG, IMG_5202.JPG, IMG_6607.JPG, and IMG_6609.JPG. For those above images in the './AlphaPilot_test2/Data_Training/' 
and corresponding ground truth in './AlphaPilot_test2/training_GT_labels_v2.json' were removed before loading. 
"""
json_data = json.load(open("C:/Users/jwang/Desktop/AlphaPilot_test2/augmented_training_GT_labels_v2.json"))

labels_number = pd.read_json(json.dumps(json_data), typ='frame', orient="index")
labels_img = pd.read_json(json.dumps(json_data), typ='series')

#print(len(labels_number))
#print(labels_number[0][1])
#print(labels_img[0][0])

data_folder = 'C:/Users/jwang/Desktop/AlphaPilot_test2/Data_Training/'

#%%
# preprocess the image data by resizing 
# preprocess the image data by changing the image color space from RGB to HSV and keep S channel only for light rubustness 
def image_preprocessing(img):
    resized = cv2.resize((cv2.cvtColor(img, cv2.COLOR_RGB2HSV)),(img_cols,img_rows))
    #gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #resized = cv2.resize(gray, (img_cols,img_rows))
    return resized
# normalize the training images
def normalized(img):
    #return (img - 128) / 128
    # normalize image data to [0.1, 0.9]
    a = 0
    b = 1.0
    minimum = 0
    maximum = 255
    return (b - a) * ((img - minimum) / (maximum - minimum)) + a 

# resize the images from 1296*864 to 96*96 
# the image dimension is tunable 
img_rows = 96
img_cols = 96
#%%
import numpy as np

# X for images as training features and y for coordinates as training labels 
X = []
y = []

N_test = 256*2
#for i in range(N_test):
for i in range(len(labels_number)):
    img_path = labels_number.index[i]
    img_path = data_folder+img_path
    print("loading images:" +img_path)
    img = plt.imread(img_path)
    X.append(image_preprocessing(img))
    
    y.append(labels_number[0][i])
print(len(y))

#plt.imshow(img)
#plt.scatter(y[-1][0:2], y[-1][3:5], s=img)
#plt.show()

#%%
# plot testing 
"""
img_test = plt.imread('C:/Users/jwang/Desktop/AlphaPilot_test2/error_data/IMG_0493.JPG')
plt.scatter(x=[729, 794, 789, 724], y=[330, 331, 492, 492], c='r', s=40)
#plt.plot([729, 330, 794, 331, 789, 492, 724, 492],'o')
plt.imshow(img_test)

#%%
img_test = plt.imread('C:/Users/jwang/Desktop/AlphaPilot_test2/Data_Training/IMG_0640.JPG')
coord = [563, 228, 729, 115, 732, 743, 558, 634]
plt.scatter(x=[coord[0], coord[2], coord[4], coord[6]], y=[coord[1], coord[3], coord[5], coord[7]], c='r', s=40)
#plt.plot([729, 330, 794, 331, 789, 492, 724, 492],'o')
plt.imshow(img_test)
"""
#%%

from sklearn.utils import shuffle

data={}
data['features'] = X
data['labels'] = y

X_train, y_train = data['features'], data['labels']

X_train = np.asarray(data['features']).astype('float32')
y_train = np.asarray(data['labels']).astype('float32')

# normalize the training label 
y_train = (y_train - 48) / 48  # scale target coordinates to [-1, 1]

# shuffle data and then split data into training and validation 
X_train, y_train = shuffle(X_train, y_train, random_state=42)

# for tensorflow 
# from sklearn.model_selection import train_test_split
#X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, random_state=0, test_size=0.1)

#%%

# reshape the training data to (None, img_rows, img_cols, 1)
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
#X_valid = X_valid.reshape(X_valid.shape[0], img_rows, img_cols, 1)

#y_train = y_train.reshape(y_train.shape[0], 8)
#y_valid = y_valid.reshape(y_valid.shape[0], 8)

#%%
X_train = normalized(X_train) 
#X_valid = normalized(X_valid)
print("Training Image Shape:     {}".format(X_train[0].shape))
print("Training Set:    {} samples".format(len(X_train)))
print()
print("Training Label Shape:     {}".format(y_train[0].shape))
print("Training Set:    {} samples".format(len(y_train)))
#print("Validation Image Shape:     {}".format(X_valid[0].shape))
#print("Validation Set:    {} samples".format(len(X_valid)))

#%%
# https://github.com/elix-tech/kaggle-facial-keypoints/blob/master/kfkd.py
# Import Keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Dropout, Activation
from tensorflow.keras.layers import Flatten, Dense

model = Sequential()

model.add(Convolution2D(32, 3, 3, input_shape=(96, 96, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Convolution2D(64, 2, 2))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Convolution2D(128, 2, 2))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Flatten())
model.add(Dense(1000))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1000))
model.add(Activation('relu'))
model.add(Dense(8))

model.summary()

#%%
"""
# this is a shallow CNN built bu keras I directly grabed from my previous project. A deeper CNN architecture
can definetely be tested, but I feel since some of the ground truth values are not very accurate made on purpose,
a shallow CNN maybe works better.  

# the model accept 96x96 pixel one chanel images in
# a fully-connected output layer with 8 values (2 for each corner)
model = Sequential()
model.add(Convolution2D(32, (5, 5), input_shape=(96,96,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Convolution2D(15, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(8))

# Summarize the model
model.summary()
"""
#%%
from tensorflow.keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam

#  Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Train the model
hist = model.fit(X_train, y_train, epochs=200, batch_size=128, verbose=1, validation_split=0.15)
# Save the model as model.h5
model.save('my_model_3c_simple2_arg.h5')
print("Trained model saved!")

#%%
# training history 
print(hist.history.keys())
# summarize history for accuracy
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#%%
# RESIZE images
import cv2
filename = 'C:/Users/JWang/Dropbox/AAA_PostDoc/AlphaPilot/RESULTES/simple_arg_2timesTraining/test10.jpg'

oriimg = plt.imread(filename)

newimg = cv2.resize(oriimg,(1296, 864))
cv2.imshow("Show by CV2",newimg)
cv2.waitKey(0)
image = cv2.imwrite('C:/Users/JWang/Dropbox/AAA_PostDoc/AlphaPilot/RESULTES/simple_arg_2timesTraining/test10.jpg',255*newimg)

#%%
# testing the trained CNN by images in './Data_LeaderboardTesting' 
from tensorflow.keras.models import load_model

img_rows = 96
img_cols = 96

model = load_model('/home/aml/Dropbox/AAA_PostDoc/AlphaPilot/RESULTES/simple_arg_2timesTraining/my_model_3c.h5')

X_test = plt.imread('/home/aml/Dropbox/AAA_PostDoc/AlphaPilot/RESULTES/simple_arg_2timesTraining/test8.jpg')

X_test = cv2.resize((cv2.cvtColor(X_test, cv2.COLOR_RGB2HSV)),(img_cols,img_rows))
#X_test = cv2.resize(cv2.cvtColor(X_test, cv2.COLOR_RGB2GRAY),(img_cols,img_rows))
X_test = X_test / 255
X_test = X_test.reshape(-1, img_rows, img_cols, 3)

y_test = model.predict(X_test)

X_org = plt.imread('/home/aml/Dropbox/AAA_PostDoc/AlphaPilot/RESULTES/simple_arg_2timesTraining/test8.jpg')

y_test = y_test * 48 + 48 # undo the normalization

plt.scatter(x=[y_test[0][0], y_test[0][2], y_test[0][4], y_test[0][6]], y=[y_test[0][1], y_test[0][3], y_test[0][5], y_test[0][7]], c='r', s=40)
plt.imshow(X_org)
#%%
# Load libraries
import json
from pprint import pprint
import glob
import cv2
import numpy as np
from random import shuffle

from helper import *
import time
import os 

import json
import cv2
import numpy as np

from tensorflow.keras import models

import pickle 

#%%
model = load_model('my_model_3c_simple2_arg.h5')
            
def img_predict(img):
    n_boxes = 1
    
    if n_boxes>0:
        bb_all = 400*np.random.uniform(size = (n_boxes,9))
        bb_all[:, 0:8] = model.predict(img)
        bb_all = bb_all * 48 + 48 
        bb_all[:,-1] = 0.5
    else:
        bb_all = []
    return bb_all.tolist()

img_file = glob.glob('Data_LeaderboardTesting/*.JPG')
img_keys = [img_i.split('/')[-1] for img_i in img_file]

# load image, convert to RGB, run model and plot detections. 
time_all = []
pred_dict = {}
for img_key in img_keys:
    img =cv2.imread(img_key)
    img_org =cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize((cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)),(96, 96))
    img = img / 255
    img = img.reshape(-1, 96, 96, 3)
       
    tic = time.monotonic()
    
    bb_all = img_predict(img)
    
    toc = time.monotonic()
    
    pred_dict[os.path.basename(img_key)] = bb_all # http://techs.studyhorror.com/python-get-last-directory-name-in-path-i-139
    time_all.append(toc-tic)
    
    
plot_bbox(img_org, bb_all)


#%%

mean_time = np.mean(time_all)
ci_time = 1.96*np.std(time_all)
freq = np.round(1/mean_time,2)
    
print('95% confidence interval for inference time is {0:.2f} +/- {1:.4f}.'.format(mean_time,ci_time))
print('Operating frequency from loading image to getting results is {0:.2f}.'.format(freq))

#%%
import json

with open('random_submission.json', 'w') as f:
    json.dump(pred_dict, f)


