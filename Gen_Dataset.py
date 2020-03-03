# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 10:16:57 2020

@author: Yashvi
"""
import cv2
import os
import csv
import glob

parent_dir="D:\\Jupyter\\cell-images-for-detecting-malaria\\"
fl = open('dataset.csv','a+')
fl.write("label")
fl.write(",")
fl.write("area1")
fl.write(",")
fl.write("area2")
fl.write(",")
fl.write("area3")
fl.write(",")
fl.write("area4")
fl.write(",")
fl.write("area5")
fl.write(",")
fl.write("\n")

for subdir,dirs,files in os.walk(parent_dir):  
    for file in files:
        label=subdir[len(parent_dir) : ]
        
        img_path=os.path.join(subdir,file)
        if img_path.endswith("png"):
            
            img= cv2.imread(img_path)
        
            blur_img = cv2.pyrMeanShiftFiltering(img,13,17)
    
            #plt_img= cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
            gry_img= cv2.cvtColor(blur_img, cv2.COLOR_BGR2GRAY)
    
            th = 120
            max_val= 255
    
            ret, op = cv2.threshold(gry_img, th, max_val, cv2.THRESH_BINARY)
    
            
            _, contours, _ = cv2.findContours(op,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
            fl.write(label)
            fl.write(",")
            for i in range(5):
                try:
                    area=cv2.contourArea(contours[i])
                    fl.write(str(area))
                except:
                    fl.write(str(0.0))
                fl.write(',')   
     
            fl.write("\n")
       
fl.close()
'''cv2.imshow("img",img) 
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
