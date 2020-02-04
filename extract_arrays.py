# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 20:02:47 2019

@author: Nuno
"""

import pandas as pd 
import cv2                 
import numpy as np         
import os                  
from random import shuffle
from tqdm import tqdm  
import scipy
import skimage
from skimage.transform import resize
import matplotlib.pyplot as plt


#Define the directories
TRAIN_DIR = ".\\cnn_sets\\train\\"
TEST_DIR =  ".\\cnn_sets\\test\\"
VAL_DIR = ".\\cnn_sets\\validation"
global_dataframe=pd.read_csv('.\\global_dataframe.csv')


#the next function was developed to extract labels and arrays by name of the folder (APPLIED ON DATASET XP2)
def get_data(Dir):
    X = []
    y = []
    for nextDir in os.listdir(Dir):
        if not nextDir.startswith('.'):
            if nextDir in ['NORMAL']:
                label = 0
            elif nextDir in ['PNEUMONIA']:
                label = 1
            else:
                label = 2
                
            temp = Dir + nextDir
                
            for file in tqdm(os.listdir(temp)):
                img = cv2.imread(temp + '/' + file)
                if img is not None:
                    img = skimage.transform.resize(img, (200, 200, 3))
                    #img_file = scipy.misc.imresize(arr=img_file, size=(150, 150, 3))
                    img = np.asarray(img)
                    X.append(img)
                    y.append(label)
                    
    X = np.asarray(X)
    y = np.asarray(y)
    return X,y

#X_train, y_train = get_data(TRAIN_DIR)
#X_test, y_test=get_data(TEST_DIR)    

#the next two functions were developed to extract labels and arrays from the same folder.
def grab_labels(global_df, img_path):
    df=global_df
    lista=[]
    for dirName, subdirList, fileList in os.walk(img_path):
        for filename in fileList:
            file=filename.split('.jpeg')
            beta=df.loc[df['patientId']== file[0]]
            x=beta['Target'].values            
            if x[0]==0:
                lista.append(0)
            else:
                lista.append(1)
    Y=np.asarray(lista)
    return Y

def grab_arrays(temp):
    X=[]
    for file in tqdm(os.listdir(temp)):
        img = cv2.imread(temp + '/' + file)
        if img is not None:
            img = skimage.transform.resize(img, (200, 200, 3))
            img = np.asarray(img)
            X.append(img)
    X = np.asarray(X)
    return X

##IT'S ADVISED TO CREATE AND SAVE EACH ARRAY AT A TIME !! HIGH MEMORY USAGE!!!

#prepare train array for pixel arrays and labels
#X_train=grab_arrays(TRAIN_DIR)
#y_train=grab_labels(global_dataframe, TRAIN_DIR)


#prepare validation array for pixel arrays and labels
#X_val=grab_arrays(VAL_DIR)
#y_val=grab_labels(global_dataframe, VAL_DIR)


#prepare test array for pixel arrays and labels
X_test=grab_arrays(TEST_DIR)
y_test=grab_labels(global_dataframe, TEST_DIR)


from keras.utils.np_utils import to_categorical


#one hot encode the label arrays.

#y_train = to_categorical(y_train, 2)
#y_val = to_categorical(y_val, 2)
y_test = to_categorical(y_test, 2)

#reshape the pixel arrays for ready use on keras.
#X_train=X_train.reshape(2020,3,400,400)
#X_val=X_val.reshape(562,3,200,200)
X_test=X_test.reshape(500,3,200,200)


#Define the output directories
#np.save('.\data_arrays\\x_train_200.npy', X_train)
#np.save('.\data_arrays\\x_val_200.npy', X_val)
np.save('.\data_arrays\\x_test_200.npy', X_test)

#np.save('.\data_arrays\\y_train_200.npy', y_train)
#np.save('.\data_arrays\\y_val_200.npy', y_val)
np.save('.\data_arrays\\y_test_200.npy', y_test)

