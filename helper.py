# helper functions
import os
import cv2
import math
import numpy as np
from skimage.measure import compare_ssim as ssim
# function to calculate the peak signal to noise ratio of low resolution and high resolution
def psnr(l_res, h_res):
    
    # convert the image data to floats
    l_resData = l_res.astype(float)
    h_resData = h_res.astype(float)
    
    # calculate the difference
    diff = h_resData - l_resData
    diff = diff.flatten('C')
    
    # calculate the root mean square difference
    rmsd = math.sqrt(np.mean(diff ** 2.))
    
    # calculate the psnr
    psnr = 20 * math.log10(255. / rmsd)
    
    return psnr

# function for mean squared error
def mse(l_res, h_res):
    
    # sum of squared differences of two images
    error = np.sum((l_res.astype(float) - h_res.astype(float)) ** 2)
    
    # divide by total number of pixels
    error /= float(l_res.shape[0] * h_res.shape[1])
    return error

# compare the qulity of low-res and high-res images
def compare_images(l_res, h_res):
    
    results = []
    results.append(psnr(l_res, h_res))
    results.append(mse(l_res, h_res))
    results.append(ssim(l_res, h_res, multichannel=True))
    
    return results


# degrade images

def degrade_images(path, value):
    print('path {}'.format(path))
    # for all the files in the given path
    for file in os.listdir(path):
        
        # read the file using cv2
        img = cv2.imread(path + '/' + file)
        
        # find the old and new image dimensions
        h, w, c = img.shape
        new_h = int(h / value)
        new_w = int(h / value)
        
        # downsize the image
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # upsize the image
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # save the image
        print('Saving {}'.format(file))
        cv2.imwrite('Test_degrded/{}'.format(file), img) 
        
        
# image pre processing
def size_mod(img, factor):
    temp_size = img.shape
    size = temp_size[0:2]
    size = size - np.mod(size, factor)
    return img[0:size[0], 1:size[1]]


def crop(img, edge):
    return img[edge:-edge, edge:-edge]        
        
    
    
def test(test_path, model):
    
    
    # load high res and and low res images
    path, file = os.path.split(test_path)
    lr = cv2.imread(test_path)
    hr = cv2.imread('Test/{}'.format(file))
    
    # take the mode of the images
    lr = size_mod(lr, 3)
    hr = size_mod(hr, 3)
    
    # convert the images to YCrCb color space
    ycrcb = cv2.cvtColor(lr, cv2.COLOR_BGR2YCrCb)
    
    # extract the Y (luminance) channel from YCrCb space
    Y = np.zeros((1, ycrcb.shape[0], ycrcb.shape[1], 1), dtype=float)
    Y[0, :, :, 0] = ycrcb[:, :, 0].astype(float) / 255
    
    # make a prediction using trained model
    prediction = model.predict(Y, batch_size=1)
    
    # post procces the images
    prediction *= 255
    prediction[prediction > 255] = 255
    prediction[prediction < 0] = 0
    prediction = prediction.astype(np.uint8)

    # reconstruct the image in BGR space
    # note the predicted image lost the 4 pixels on each side therefore we need the crop
    # the image with a factor of 4
    ycrcb = crop(ycrcb, 4)
    ycrcb[:, :, 0] = prediction[0, :, :, 0] 
    recon_image = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    
    # remove the border of the lr and hr image for comparison
    lr = crop(lr.astype(np.uint8), 4)
    hr = crop(hr.astype(np.uint8), 4)
    
    # image comparison
    metrics = []
    metrics.append(compare_images(lr, hr))
    metrics.append(compare_images(recon_image, hr))
    
    # return hr, lr, reconstructed image and metrics
    return hr, lr, recon_image, metrics