import cv2
import numpy as np
import math
from skimage.measure import shannon_entropy


def contrast(img):
    #single
    return img.var()


def entropy(img):
    return shannon_entropy(img,base=math.e)


def aveGradient(img):
    height=img.shape[0]
    width=img.shape[1]
    gradient=0
    for i in range(height-1):
        for j in range(width-1):
            temp=(((img[i,j+1]-img[i,j])**2)+((img[i+1,j]-img[i,j])**2))/2
            gradient+=math.sqrt(temp)

    return gradient/((height-1)*(width-1))

#return all metrics and 3 tongdao
def get_all_metrics(img):
    img_g=img[:,:,0]
    img_b = img[:, :, 1]
    img_r = img[:, :, 2]

    #contrast
    contrast_g=contrast(img_g)
    contrast_b = contrast(img_b)
    contrast_r = contrast(img_r)
    contrast_ave=(contrast_g+contrast_b+contrast_r)/3
    #entropy
    entropy_g = entropy(img_g)
    entropy_b = entropy(img_b)
    entropy_r = entropy(img_r)
    entropy_ave = (entropy_g+entropy_b+entropy_r)/3

    aveGradient_g=aveGradient(img_g)
    aveGradient_b = aveGradient(img_b)
    aveGradient_r = aveGradient(img_r)
    aveGradient_ave = (aveGradient_g+aveGradient_b+aveGradient_r)/3

    return contrast_ave,entropy_ave,aveGradient_ave

def SNR(img):
    pass
