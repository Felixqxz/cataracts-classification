#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 14:28:28 2018

@author: Felixsmacbook
"""

from keras.datasets import mnist
# %% load the dataset
(x_train, y_train),(x_test, y_test) = mnist.load_data()
type(x_train)
print(x_train.shape, y_train.shape)
x_train_0 = x_train[0]
y_train_0 = y_train[0]
import matplotlib.pyplot as plt
plt.imshow(x_train_0)
plt.title('Training Label ()'.format(y_train_0))
plt.show()
y_train.min()
y_train.max()