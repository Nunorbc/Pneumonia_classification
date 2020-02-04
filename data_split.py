# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 00:28:51 2019

@author: Nuno
"""

import os
import shutil
import pandas as pd
import pydicom as pdicom
from PIL import Image
import random


#Separate by class: only select Normal and Pneumonia. "NotNormal/NotPneumonia is discarded".
def identification(global_df, training_info, all_images_path, destinatary):
    df=pd.read_csv(global_df)
    df2=pd.read_csv(training_info)
    for dirName, subdirList, fileList in os.walk(all_images_path):
        for filenameDCM in fileList:
            file=filenameDCM.split('.dcm')
#            beta==df.loc[df['patientId']== file[0]]
#            v=beta['Target'].values
            beta2=df2.loc[df2['patientId']==file[0]]
            v2=beta2['class'].values
            beta3=pdicom.read_file(dirName+filenameDCM)
            v3=beta3.ViewPosition
            
            if v3 == 'PA':
                if v2[0]=='Normal' or v2[0]=='Lung Opacity':
                    print(v2[0], v3)
                    print('true')
                    shutil.move(all_images_path+filenameDCM,destinatary+filenameDCM) #folder of only Normal and Pneumonia images
                
#identification('.\stage_2_train_labels.csv',
#               '.\stage_2_detailed_class_info.csv',
#               '.\all_images\\',
#               '.\normal_lungopacity\\')                    


#separate the selected images between Normal and Pneumonia. ONE WAY
def separate_normal_pneumonia(training_info, all_images_path, destinatary):
    df2=pd.read_csv(training_info)
    for dirName, subdirList, fileList in os.walk(all_images_path):
        for filenamejpeg in fileList:
            file=filenamejpeg.split('.jpeg')
#            beta==df.loc[df['patientId']== file[0]]
#            v=beta['Target'].values
            beta2=df2.loc[df2['patientId']==file[0]]
            v2=beta2['class'].values           
            if v2[0]=='Lung Opacity':
                print(v2[0])
                print('true')
                shutil.move(all_images_path+filenamejpeg,destinatary+filenamejpeg)
                
#separate_normal_pneumonia('.\stage_2_detailed_class_info.csv'
#                          ,'.\normal_lungopacity\\train\\'
#                          ,'.\PNEUMONIA\\')


#Convert all the normal_pneumonia folder images into JPEG.                                
def convert_to_jpeg(training_df,folder,destinatary):
    df=pd.read_csv(training_df)
    for dirName, subdirList, fileList in os.walk(folder):
        for filenameDCM in fileList:
            file=filenameDCM.split('.dcm')
            beta2=df.loc[df['patientId']==file[0]]
            v2=beta2['Target'].values
            
            ds=pdicom.read_file(dirName+ filenameDCM)
            ArrayDicom = ds.pixel_array
#            print(ArrayDicom)
            ID=ds.PatientID
            filen=destinatary+ '/' + ID
            im = Image.fromarray(ArrayDicom)
            im.save(filen+ '.jpeg')
                
#convert_to_jpeg('.\stage_2_train_labels.csv','.\normal_lungopacity\\',
#                '.\normal_lungopacity_jpeg\\')
                
import os, random

##choose the train, validation and test number or percentage and random move the selected images to other folder.

direct=".\normal_lungopacity_jpeg\\"
for i in range(0,500):
    x=random.choice(os.listdir(direct))
    shutil.move(direct+'/'+x, '.\cnn_sets\\test')

for i in range(0,1562):
    x=random.choice(os.listdir(direct))
    shutil.move(direct+'/'+x, '.\cnn_sets\\validation') 

for i in range(0,6500):
    x=random.choice(os.listdir(direct))
    shutil.move(direct+'/'+x, '.\cnn_sets\\train')               