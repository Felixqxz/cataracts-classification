#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 16:22:16 2018

@author: Felixsmacbook
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage import io,transform
import tensorflow as tf
from keras.layers import Input, Dense, Activation
from keras.models import Model
from keras.layers import Conv2D, Flatten, MaxPool2D
from keras.optimizers import SGD, RMSprop, adam
from keras.utils import to_categorical
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte
from skimage import color
from keras.preprocessing import image
from keras.utils import generic_utils
from keras.callbacks import TensorBoard

w = 224
h = 224
c = 3
path = '/Users/Philixsmacbook/Desktop/Images'

imgs = []
label = []
count = 0
for train_class in os.listdir(path):
    for pic in os.listdir(path+'/'+train_class):
        img = io.imread(path+'/'+train_class+'/'+pic)
        img = transform.resize(img,(w,h,c))
        imgs.append(img)
        label.append(train_class)
        temp = np.array([imgs,label])
        temp = temp.transpose()
        np.random.shuffle(temp)
        image_list = list(temp[:,0])
        label_list = list(temp[:,1])
        count += 1
#print (count)

nb_images = int(0.8*count)
x_train = np.array(image_list[:nb_images])
y_train = np.array(label_list[:nb_images])

nb_test = int(0.2*count)
x_test = np.array(image_list[:nb_test])
y_test = np.array(label_list[:nb_test])
print (x_test, y_train)

from keras.layers import Convolution2D, MaxPooling2D, concatenate, Dropout
from keras.layers import GlobalAveragePooling2D

sq1x1 = "squeeze1x1"
exp1x1 = "expand1x1"
exp3x3 = "expand3x3"
relu = "relu_"

def fire_module(x, fire_id, squeeze=16, expand=64):
   s_id = 'fire' + str(fire_id) + '/'
   x = Convolution2D(squeeze, (1, 1), padding='valid', name=s_id + sq1x1)(x)
   x = Activation('relu', name=s_id + relu + sq1x1)(x)

   left = Convolution2D(expand, (1, 1), padding='valid', name=s_id + exp1x1)(x)
   left = Activation('relu', name=s_id + relu + exp1x1)(left)

   right = Convolution2D(expand, (3, 3), padding='same', name=s_id + exp3x3)(x)
   right = Activation('relu', name=s_id + relu + exp3x3)(right)

   x = concatenate([left, right], axis=3, name=s_id + 'concat')
   return x

def SqueezeNet():
    inputs = Input(shape=(224, 224, 3))
    x = Convolution2D(64, (3, 3), strides=(2, 2), padding='valid', name='conv1')(inputs)
    x = Activation('relu', name='relu_conv1')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)
    
    x = fire_module(x, fire_id=2, squeeze=16, expand=64)
    x = fire_module(x, fire_id=3, squeeze=16, expand=64)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)
    
    x = fire_module(x, fire_id=4, squeeze=32, expand=128)
    x = fire_module(x, fire_id=5, squeeze=32, expand=128)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)
    
    x = fire_module(x, fire_id=6, squeeze=48, expand=192)
    x = fire_module(x, fire_id=7, squeeze=48, expand=192)
    x = fire_module(x, fire_id=8, squeeze=64, expand=256)
    x = fire_module(x, fire_id=9, squeeze=64, expand=256)
    x = Dropout(0.5, name='drop9')(x)
    
    x = Convolution2D(4, (1, 1), padding='valid', name='conv10')(x)
    x = Activation('relu', name='relu_conv10')(x)
    x = GlobalAveragePooling2D()(x)
    out = Activation('softmax', name='loss')(x)
    
    model = Model(inputs, out, name='squeezenet')
    return model

model = SqueezeNet()

#
#img = image.load_img('pexels-photo-280207.jpeg', target_size=(227, 227))
#x = image.img_to_array(img)
#x = np.expand_dims(x, axis=0)
#x = preprocess_input(x)
#

#all_results = decode_predictions(preds)
#for results in all_results:
#  for result in results:
#    print('Probability %0.2f%% => [%s]' % (100*result[2], result[1]))
#    

#        
model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
#
y_train_prob = to_categorical(y_train)
#print(y_train_small_prob)
y_test_prob = to_categorical(y_test)

epochs = 15
batch_size = 32
log_dir = 'logs/'
tb_callback = TensorBoard(log_dir, histogram_freq=1, write_graph=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
img_generator = ImageDataGenerator(rotation_range = 90,
                              width_shift_range = 0.2,
                              height_shift_range = 0.2,
                              zoom_range = 0.3)

img_generator.fit(x_train)

model.fit_generator(img_generator.flow(x_train, y_train_prob, batch_size=batch_size),
                    steps_per_epoch=len(x_train), epochs=epochs, validation_data = (x_test, y_test_prob), callbacks=[tb_callback])

#for e in range(4*epochs):
#    print('Epoch', e)
#    print('Training...')
#    progbar = generic_utils.Progbar(x_train.shape[0])
#    batches = 0
#
#    for x_batch, y_batch in img_generator.flow(x_train, y_train_prob, batch_size=batch_size, shuffle=True):
#        loss, train_acc = model.train_on_batch(x_batch, y_batch)
#        batches += 1
#        if batches > len(x_train)/32:
#            break
#        progbar.add(x_batch.shape[0], values=[('train loss', loss),('train acc', train_acc)])
##        
#test_img = image.load_img('/Users/Philixsmacbook/Desktop/Images/2/C2N1P2(2).png', target_size=(224, 224))
#x = image.img_to_array(test_img)
#x = np.expand_dims(x, axis=0)
#
#nb_epochs=10
#model.fit_generator(x_train, y_train_prob, epochs=nb_epochs,
#validation_data = (x_test, y_test_prob))
#print (y_train_small_prob.shape)
#preds = model.predict(x)
#print (preds)