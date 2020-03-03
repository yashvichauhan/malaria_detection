# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 22:43:29 2019

@author: Yashvi
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
import joblib
import cv2

def image_proc(img_path):
    img= cv2.imread(img_path,cv2.IMREAD_UNCHANGED)
    
    img_lst=[]
    plt_img= cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    plt.imshow(plt_img)
    plt.title("original image")
    plt.show()
    
    blur_img = cv2.pyrMeanShiftFiltering(img,13,17)
    
    #grayimg
    gry_img= cv2.cvtColor(blur_img, cv2.COLOR_RGB2GRAY)
    
    th = 120
    max_val= 255
    
    ret, op = cv2.threshold(gry_img, th, max_val, cv2.THRESH_BINARY)
    
    #find contour        
    _, contours, _ = cv2.findContours(op,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #enter data into list
    for x in range(5):
        try:
            area=cv2.contourArea(contours[x])
            img_lst.append(str(area))
        except:
            img_lst.append(str(0.0))
           
    #print(img_lst)
    return img_lst

#classifying images

df = pd.read_csv('D:\Python\datasetC.csv')
print(df)
x = df.drop(['label'],axis=1)
y = df['label']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=4)
#classify
model = RandomForestClassifier(n_estimators=300,max_depth=10)
#fit model
model.fit(x_train,y_train)
#save model 
joblib.dump(model,"malaria_rf_300_10")
#predict
prediction = model.predict(x_test)

print(metrics.classification_report(prediction,y_test))

#creating one prediction on any random image path

test_img_path="D:\\Jupyter\\cell-images-for-detecting-malaria\\Parasitized\\C33P1thinF_IMG_20150619_115740a_cell_162.png"
inp=image_proc(test_img_path)
inp1=[[None]*5]*1
for x in range(5):
    inp1[0][x]=inp[x]
#data
print("confusion matrix\n")
print(metrics.confusion_matrix(prediction,y_test))
print('\n')
#predict
mj= joblib.load('malaria_rf_100_5')

print("Test result:"+ mj.predict(inp1))
print(metrics.accuracy_score())
