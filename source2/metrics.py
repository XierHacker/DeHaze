import cv2
import numpy as np
import math
from skimage.measure import shannon_entropy

def contrast(img):
    #single
    return img.var()


def entropy(img):
    return shannon_entropy(img,base=math.e)
    '''
    stat=np.zeros(shape=(256,))
    H=0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            stat[img[i,j]]+=1
    #prob
    stat=stat/(img.shape[0]*img.shape[1])

    for index in range(stat.shape[0]):
        if(stat[index]!=0):
            H+=stat[index]*np.log(stat[index])
    return -1*H
    '''


def aveGradient(img):
    pass

def SNR(img):
