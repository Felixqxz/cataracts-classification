#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 15:42:32 2018

@author: Felixsmacbook
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io,transform
from keras.layers import Input, Activation
from keras.models import Model
from keras.utils import to_categorical
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.layers import Convolution2D, MaxPooling2D, concatenate, Dropout
from keras.layers import GlobalAveragePooling2D

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
        count += 1
        
temp = np.array([imgs,label])
temp = temp.transpose()
np.random.shuffle(temp)
image_list = list(temp[:,0])
label_list = list(temp[:,1])    

nb_images = int(0.8*count)
x_train = np.array(image_list[:nb_images])
y_train = np.array(label_list[:nb_images])

nb_test = int(0.2*count)
x_test = np.array(image_list[:-nb_test])
y_test = np.array(label_list[:-nb_test])
print (x_test, y_train)

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

model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])

y_train_prob = to_categorical(y_train)
#print(y_train_small_prob)
y_test_prob = to_categorical(y_test)

epochs = 20
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

test_path = '/Users/Philixsmacbook/Desktop/Images/2/C1N0P4.png'
test_img = image.load_img(test_path, target_size=(224, 224))
x = image.img_to_array(test_img)
x = np.expand_dims(x, axis=0)
preds = model.predict(x).tolist()    
if (preds[0][0] > preds[0][1] and preds[0][0] > preds[0][2] and preds[0][0] > preds[0][3]):
    print ("Cortical")
if (preds[0][1] > preds[0][0] and preds[0][1] > preds[0][2] and preds[0][1] > preds[0][3]):
    print ("Nuclear")
if (preds[0][2] > preds[0][0] and preds[0][2] > preds[0][1] and preds[0][2] > preds[0][3]):
    print ("Subcapsular")
if (preds[0][3] > preds[0][0] and preds[0][3] > preds[0][1] and preds[0][3] > preds[0][2]):
    print ("Transparent")