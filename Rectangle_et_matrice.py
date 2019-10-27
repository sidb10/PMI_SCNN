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
import scipy.io as spio



myPATH='C:/Users/sebastien/Desktop/PMI/polar_car_set/Annotations/snc00004.mat'
matrix='snc00004_crop'
f = h5py.File(myPATH,'r')
data = f.get(matrix)
data = np.array(data) 

print(data)

imgpil=PIL.Image.open('C:/Users/sebastien/Desktop/PMI/polar_car_set/polar_car_set/snc00004/0.jpg','r')

im = np.array(imgpil)

# Create figure and axes
fig,ax = plt.subplots(1)

# Display the image
ax.imshow(im)

# Create a Rectangle patch
# Add the patch to the Axes
for i in range(0,data.shape[1]):
    
    rect=patches.Rectangle((data[0,i],data[1,i]),data[2,i],data[3,i],  linewidth=1 ,edgecolor='r',facecolor='none')
    ax.add_patch(rect)



