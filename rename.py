#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 09:26:54 2018

@author: Felixsmacbook
"""

import os

path = '/Users/Philixsmacbook/Desktop/Images/cortical'
i = 1
for file in os.listdir(path):
    if os.path.isfile(os.path.join(path,file))==True:
        new_name=file.replace(file,"%d.tif"%i)
        os.rename(os.path.join(path,file),os.path.join(path,new_name))
        i+=1
        