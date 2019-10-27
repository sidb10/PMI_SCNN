# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 18:50:38 2019

"""



import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import h5py
import  PIL
import scipy.io as spio
from os import listdir, path
from os.path import isfile, join


dataFiles = [f for f in listdir('C:/Rcode/Python/Pmi/polar_car_set/Annotations/') if isfile(join('C:/Rcode/Python/Pmi/polar_car_set/Annotations/', f))]

dataImages=[f for f in listdir('C:/Rcode/Python/Pmi/polar_car_set/polar_car_set/')]




for j in range(0,5):#max= 144
    imgpil=PIL.Image.open('C:/Rcode/Python/Pmi/polar_car_set/polar_car_set/{0}/0.jpg'.format(dataImages[j]),'r')
    im = np.array(imgpil)
    # Create figure and axes
    fig,ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)        
    myPATH='C:/Rcode/Python/Pmi/polar_car_set/Annotations/{0}'.format(dataFiles[j])      
    matrix='{0}_crop'.format(dataFiles[j][0:-4])
    f = h5py.File(myPATH,'r')
    data = f.get(matrix)
    data = np.array(data) 
    data=data.T
    for k in range(0,data.shape[0]):
        rect=patches.Rectangle((data[k,0],data[k,1]),data[k,2],data[k,3],  linewidth=1 ,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
