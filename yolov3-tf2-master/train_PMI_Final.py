# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 15:35:38 2020

@author: Sidney Bakhouche
"""


# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 11:59:05 2020

@author: Sidney Bakhouche
"""
from absl import app, flags, logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import h5py
import  PIL
import scipy.io as spio
from os import listdir, path
from os.path import isfile, join
import cv2
from yolov3_tf2.dataset import transform_images
import tensorflow as tf
import os
import cv2
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard
)
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny, YoloLoss,
    yolo_anchors, yolo_anchor_masks,
    yolo_tiny_anchors, yolo_tiny_anchor_masks
)
from yolov3_tf2.utils import freeze_all
import yolov3_tf2.dataset as dataset


dataFiles = [f for f in listdir('./data/polar_car_set/Annotations/') if isfile(join('./data/polar_car_set/Annotations/', f))]

dataImages=[f for f in listdir('./data/polar_car_set/Images/')]



def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_xy(data,shape):
    h,w=shape
    for i in range(0,len(data)):
        xmin=data[i][0]/w
        ymin=data[i][1]/h
        xmax=(data[i][0]+data[i][2])/w
        ymax=(data[i][1]+data[i][3])/h
        data[i]=xmin,ymin,xmax,ymax
    return data

def getwh(im):
    h,w=im.shape[0],im.shape[1]
    return(h,w)
    

images=[]
for j in range(0,len(dataImages)):#max= 144
    imgpil=PIL.Image.open('./data/polar_car_set/Images/{0}/0.jpg'.format(dataImages[j]),'r')
    im = np.array(imgpil)
    images.append(im)        
datas=[]
labels={}
for j in range(0,len(dataFiles)):
    label=[]
    myPATH='./data/polar_car_set/Annotations/{0}'.format(dataFiles[j])      
    matrix='{0}_crop'.format(dataFiles[j][0:-4])
    f = h5py.File(myPATH,'r')
    data = f.get(matrix)
    data = np.array(data)
    data=data[0:4]
    data=data.T
    data=convert_xy(data,getwh(images[j]))
    data=np.c_[data,np.zeros(len(data))]
    # h,w=getwh(images[j])
    tmp=[]
    labels[j]=data
    # for i in range(0,len(data)):
        
    #     label=np.asarray(convert((w,h),data[i]))
    #     tmp.append(label)
    #     temp=np.c_[tmp,np.zeros(len(tmp))]  
    #     labels[j]=temp
    # datas.append(data)

def from_yolo_to_cor(box, shape):
    img_h, img_w, _ = shape
    # x1, y1 = ((x + witdth)/2)*img_width, ((y + height)/2)*img_height
    # x2, y2 = ((x - witdth)/2)*img_width, ((y - height)/2)*img_height
    x1, y1 = int((box[0] + box[2]/2)*img_w), int((box[1] + box[3]/2)*img_h)
    x2, y2 = int((box[0] - box[2]/2)*img_w), int((box[1] - box[3]/2)*img_h)
    return x1, y1, x2, y2
    
def draw_boxes(img, boxes,shape):
    for box in boxes:
        x1, y1, x2, y2 = from_yolo_to_cor(box, shape)
        print(x1,y1,x2,y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 3)
    plt.imshow(img)
    return
    

dataImages=[f for f in listdir('./data/polar_car_set/Images')]
polar=['0','45','90']
data9chan=[]
for j in range(len(dataImages)):
    tmp=[]
    for l in range(len(polar)):    
        
        img = tf.image.decode_image(open('./data/polar_car_set/Images/{0}/{1}.jpg'.format(dataImages[j],
                                         polar[l]), 'rb').read(), channels=3)        

        img = transform_images(img, 416)

        tmp.append(img)
    tmp9chan=np.concatenate((tmp[0],tmp[1],tmp[2]),axis=2)
    data9chan.append(tmp9chan)

data_list=data9chan
   
data_Array=np.asarray(data9chan,dtype='float32')
data_Array=data_Array.reshape(data_Array.shape[0],data_Array.shape[1],data_Array.shape[2]*3,3)


nb_max_box=100
list_labels=[]
for i in labels.values():
    i=np.concatenate((i,np.zeros((nb_max_box-i.shape[0],5)))).astype("float32")
    list_labels.append(i)

x_train=data_Array[0:122]
x_val=data_Array[122:137]
x_test=data_Array[137:152]

y_train=list_labels[0:122]
y_val=list_labels[122:137]
y_test=list_labels[137:152]


model = YoloV3(832, channels=3,training=True, classes=2)


anchors = yolo_anchors
anchor_masks = yolo_anchor_masks

# train_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train)) 
# train_dataset = train_dataset.shuffle(buffer_size=512)

class_names = [c.strip() for c in open('./data/car.names').readlines()]
logging.info('classes loaded')

train_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train)) 
train_dataset = train_dataset.shuffle(buffer_size=128)

###############Uncomment to visualise dataset

# from yolov3_tf2.utils import draw_outputs
# for image, labels in train_dataset.take(1):
#     boxes = []
#     scores = []
#     classes = []
#     for x1, y1, x2, y2, label in labels:
#         if x1 == 0 and x2 == 0:
#             continue

#         boxes.append((x1, y1, x2, y2))
#         scores.append(1)
#         classes.append(label)
#     nums = [len(boxes)]
#     boxes = [boxes]
#     scores = [scores]
#     classes = [classes]

#     logging.info('labels:')
#     for i in range(nums[0]):
#         logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
#                                            np.array(scores[0][i]),
#                                            np.array(boxes[0][i])))

#     img = cv2.cvtColor(image.numpy(), cv2.COLOR_RGB2BGR)
#     img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
#     plt.imshow(img)
#     cv2.imwrite('./OutputTEST.jpg', img)
#     logging.info('output saved to: {}'.format('./OutputTEST.jpg'))
###############


train_dataset = train_dataset.batch(1)
train_dataset = train_dataset.map(lambda x, y: ( tf.image.resize(x, (832, 832)),
        dataset.transform_targets(y, anchors, anchor_masks, 832)))
train_dataset = train_dataset.prefetch(
    buffer_size=tf.data.experimental.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((x_val,y_val)) 
val_dataset = val_dataset.batch(1)
val_dataset = val_dataset.map(lambda x, y: ( tf.image.resize(x, (832, 832)),
        dataset.transform_targets(y, anchors, anchor_masks, 832)))
val_dataset = val_dataset.prefetch(
    buffer_size=tf.data.experimental.AUTOTUNE)
    

model_pretrained = YoloV3(416, training=True, classes=80)
model_pretrained.load_weights('./checkpoints/yolov3.tf')


model.get_layer('yolo_darknet').set_weights(
model_pretrained.get_layer('yolo_darknet').get_weights())
freeze_all(model.get_layer('yolo_darknet'))

optimizer = tf.keras.optimizers.Adam(lr=1e-4)
loss = [YoloLoss(anchors[mask], classes=2)
        for mask in anchor_masks]
model.compile(optimizer=optimizer, loss=loss)

callbacks = [
    ReduceLROnPlateau(verbose=1),
    EarlyStopping(patience=3, verbose=1),
    ModelCheckpoint('checkpoints/yolov3_train_{epoch}.tf',
                    verbose=1, save_weights_only=True),
    TensorBoard(log_dir='logs')
]

history = model.fit(train_dataset,
                    epochs=30,
                    callbacks=callbacks,
                    validation_data=val_dataset)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.title('Loss Function')
plt.show()



