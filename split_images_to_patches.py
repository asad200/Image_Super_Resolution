# image pre processing, split to f*f patches, extract Luminance channel and prepare train data labels

import os
import cv2
import numpy as np
from helper import crop

def image_split(path, FACTOR, PATCH_SIZE, STRIDE):
    
    x_train = []
    y_train = []
    for i, file in enumerate(os.listdir(path)):
        
        # read the file using cv2
        hr = cv2.imread(path + '/' + file)
        
        # find the old and new image dimensions
        h, w, c = hr.shape
        
        # change the image color channel to YCrCb
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2YCrCb)
        hr = hr[:, :, 0]
       
        
        
        
        # degrade the images by downsizing and upsizing
        new_h = int(h / FACTOR)
        new_w = int(w / FACTOR) 
        lr = cv2.resize(hr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        lr = cv2.resize(lr, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # number of stride steps
        w_steps = int((w -(PATCH_SIZE - STRIDE)) / STRIDE)
        h_steps = int((h -(PATCH_SIZE - STRIDE)) / STRIDE)
        
        #print('w: {}'.format(w))
        #print('h: {}'.format(h))
        #print('w_steps: {}'.format(w_steps))
        #print('h_steps: {}'.format(h_steps))
        
        hr = hr.astype(float) / 255
        lr = lr.astype(float) / 255
        
        for i in range(w_steps):
            for j in range(h_steps):
                
                hr_patch = hr[j * STRIDE: j * STRIDE + PATCH_SIZE , i * STRIDE: i * STRIDE + PATCH_SIZE]
                lr_patch = lr[j * STRIDE: j * STRIDE + PATCH_SIZE , i * STRIDE: i * STRIDE + PATCH_SIZE]
                
                if hr_patch.shape[0] == hr_patch.shape[1]:
                    x_train.append(lr_patch)
                    y_train.append(crop(hr_patch, 4)) 
                    
    x_train = np.array(x_train, dtype=float)
    y_train = np.array(y_train, dtype=float)
    
    return x_train[...,np.newaxis], y_train[...,np.newaxis]