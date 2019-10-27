# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 11:32:03 2019

@author: Sidney Bakhouche
"""


import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import h5py
import  PIL

myPATH='C:/Rcode/Python/Pmi/polar_car_set/Annotations/snc00001.mat'
matrix='snc00001_crop'
f = h5py.File(myPATH,'r')
data = f.get(matrix)
data = np.array(data) 

imgpil=PIL.Image.open('C:/Rcode/Python/Pmi/polar_car_set/polar_car_set/snc00001/0.jpg','r')

im = np.array(imgpil)

# Create figure and axes
fig,ax = plt.subplots(1)

# Display the image
ax.imshow(im)

# Create a Rectangle patch
rect = patches.Rectangle((389,45),239,221   ,linewidth=1,edgecolor='r',facecolor='none')
rect1 = patches.Rectangle((6,55),323,209   ,linewidth=1,edgecolor='r',facecolor='none')

# Add the patch to the Axes
ax.add_patch(rect)
ax.add_patch(rect1)


plt.show()
