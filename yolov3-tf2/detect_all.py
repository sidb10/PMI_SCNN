# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 17:46:26 2019

@author: Sidney Bakhouche
"""

import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
    )
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs
from os import listdir, path
from os.path import isfile, join

#dataImages=[f for f in listdir('C:/Rcode/Python/Pmi/polar_car_set/polar_car_set/')]
#
#for j in range(0,2):
#    imgpil=PIL.Image.open('C:/Rcode/Python/Pmi/polar_car_set/polar_car_set/{0}/0.jpg'.format(dataImages[j]),'r')
#    im = np.array(imgpil)
#    runfile('E:/Python/yolov3-tf2/detect.py',args='-image ./data/0.jpg', wdir='E:/Python/yolov3-tf2')
#    
#    
def main(_argv):
    
    yolo = YoloV3(classes=80)
    
    yolo.load_weights('./checkpoints/yolov3.tf')
    logging.info('weights loaded')

    class_names = [c.strip() for c in open('./data/coco.names').readlines()]
    logging.info('classes loaded')
    dataImages=[f for f in listdir('./data/polar_car_set/')]
    polar=['0','45','90']
    for j in range(len(dataImages)):
        for l in range(len(polar)):    
            img = tf.image.decode_image(open('./data/polar_car_set/{0}/{1}.jpg'.format(dataImages[j],polar[l]), 'rb').read(), channels=3)
            img = tf.expand_dims(img, 0)
            img = transform_images(img, 416)
        
            t1 = time.time()
            boxes, scores, classes, nums = yolo(img)
            t2 = time.time()
            logging.info('time: {}'.format(t2 - t1))
        
            logging.info('detections:')
            for i in range(nums[0]):
                logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                                   np.array(scores[0][i]),
                                                   np.array(boxes[0][i])))
        
            img = cv2.imread('./data/polar_car_set/{0}/{1}.jpg'.format(dataImages[j],polar[l]))
            img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
            cv2.imwrite('E:/python/yolov3-tf2/output/polar_car_set/{0}/{1}.jpg'.format(dataImages[j],polar[l]), img)
            logging.info('output saved to: E:/python/yolov3-tf2/output/polar_car_set/{0}/{1}.jpg'.format(dataImages[j],polar[l]))

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass