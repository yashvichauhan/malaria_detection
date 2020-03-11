# -*- coding: utf-8 -*-
"""

@author: Yashvi
"""

from keras.layers import Input,Lambda, Dense, Flatten
from keras.models import Model
#from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt

IMAGE_SIZE=[224,224]

train_path='D:\\Jupyter\\cell-images\\train'
test_path='D:\\Jupyter\\cell-images\\test'

vgg= VGG19(input_shape=IMAGE_SIZE+ [3], weights='imagenet',include_top=False)
#for already trained layer 
for layer in vgg.layers:
    layer.trainable=False
    
folders=['D:\\Jupyter\\cell-images\\train\\Parasitized','D:\\Jupyter\\cell-images\\train\\Uninfected']
#faltten output layer into single layer
x=Flatten()(vgg.output)
#activation layer
prediction= Dense(len(folders),activation='softmax')(x)

model=Model(inputs=vgg.input, outputs=prediction)
#taking model summary 
print(model.summary())

#cost and optimization
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

#importing image from dataset

train_datagen= ImageDataGenerator(rescale=1./255,
                                  shear_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True)

test_datagen=ImageDataGenerator(rescale=1./255)

training_set= train_datagen.flow_from_directory('D:\\Jupyter\\cell-images\\train',
                                                target_size=(224,224),
                                                batch_size=32,
                                                class_mode='categorical')

test_set= test_datagen.flow_from_directory('D:\\Jupyter\\cell-images\\test',
                                           target_size=(224,224),
                                           batch_size= 32,
                                           class_mode='categorical')
#fitting the model
r= model.fit_generator(training_set,
                       validation_data=test_set,
                       epochs=15,
                       steps_per_epoch=len(training_set),
                       validation_steps=len(test_set))  

#loss

plt.plot(r.history['loss'],label='train_loss')
plt.plot(r.history['val_loss'],label='val_loss')
plt.legend()
plt.show()
plt.savefig('lossval_loss')

#accuracies

plt.plot(r.history['acc'],label='train_acc')
plt.plot(r.history['val_acc'],label='val_acc')
plt.legend()
plt.show()
plt.savefig('accval_acc')

