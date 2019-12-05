# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 10:26:44 2019

@author: Sidney Bakhouche
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import h5py
import  PIL
import scipy.io as spio
from os import listdir, path
from os.path import isfile, join
import tensorflow as tf
from yolov3_tf2.dataset import transform_images


dataImages=[f for f in listdir('./data/polar_car_set/')]
polar=['0','45','90']
data9chan=[]
for j in range(len(dataImages)):
    tmp=[]
    for l in range(len(polar)):    
        
        img = tf.image.decode_image(open('./data/polar_car_set/{0}/{1}.jpg'.format(dataImages[j],
                                         polar[l]), 'rb').read(), channels=3)        
#        fig,ax = plt.subplots(1)
#        ax.imshow(img)
#        plt.show()
        img = transform_images(img, 416)
#        fig,ax = plt.subplots(1)
#        ax.imshow(img)
#        plt.show()
        tmp.append(img)
    tmp9chan=np.concatenate((tmp[0],tmp[1],tmp[2]),axis=2)
    data9chan.append(tmp9chan)
