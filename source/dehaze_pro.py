from __future__ import print_function,division
import numpy as np
import cv2


#get the min channel map
def getMinMap(img):
    #print ("shape of this image:",img.shape)
    minMap=img.min(axis=2)
    return minMap

#get dark channel map
def getDarkMap(img,pitchSize=9):
    darkMap=np.zeros(shape=(img.shape[0],img.shape[1]),dtype=np.uint8)
    #padding,and darkMap has the same shape with the img
   # print ("shape of darkmap",darkMap.shape[0],darkMap.shape[1])
    padSize=(pitchSize-1)//2
    #print ("type of pitchsize",type(padSize))
    pad=cv2.copyMakeBorder(img,padSize,padSize,padSize,padSize,cv2.BORDER_CONSTANT,value=255)
    for i in range(darkMap.shape[0]):
        for j in range(darkMap.shape[1]):
            darkMap[i,j]=pad[i:i+pitchSize,j:j+pitchSize].min()

    return pad,darkMap

def getAtmosphericLight(img,darkMap,percent=0.001,kind=1):
    if kind == 1:
        #qulity param of water
        c=2.190
        '''
        VolumeScatterList=[
            2.936,2.936,2.936,2.936,2.936,2.936,2.936,2.936,2.936,2.936,2.936,
            2.935,2.935,2.935,2.933,2.932,2.930,2.926,2.920,2.936,2.936,2.936]
        '''

        return (1.06*8.41*200/c)

    if kind == 2:
        return None
    if kind == 3:
        return None
    if kind == 4:
        return None

    '''
    amout=int(darkMap.shape[0]*darkMap.shape[1]*percent)
    darkMap_flat=darkMap.flatten()
    #sort
    darkMap_flat.sort()
    #reverse
    darkMap_flat=darkMap_flat[::-1]
    #when the number of pixel less than 1,we use the biggest values as A
    if amout==0:
        A=darkMap_flat[0]
    else:
        A=int(darkMap_flat[0:amout].mean())
    return A
    '''




def getTransmissionMap(darkMap,atmosphericLight,omega=0.95):
    transmission=1-(omega*darkMap)/atmosphericLight
    return transmission

def getRecoverMap(img,transMap,atmosphericLight):
    #t>=0.1
    for i in range(0,transMap.shape[0]):
        for j in range(0,transMap.shape[1]):
            if transMap[i,j]<0.1:
                transMap[i,j]=0.1

    J=np.zeros(shape=img.shape)
    for i in range(0,3):
        J[:,:,i]=img[:,:,i]
        J[:,:,i]=(J[:,:,i]-atmosphericLight)/transMap+atmosphericLight

    for i in range(J.shape[0]):
        for j in range(J.shape[1]):
            for k in range(J.shape[2]):
                if J[i,j,k]>255:
                    J[i, j, k] = 255
                if J[i,j,k]<0:
                    J[i, j, k] = 0
    return np.uint8(J)