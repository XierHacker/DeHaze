import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import dehaze3
import math

def drawProb(folderPath):
    file_list=os.listdir(folderPath)
    file_nums=len(file_list)
    #dict to get pixel light info of dark map
    result=np.zeros(shape=(256,))
    accum=np.zeros(shape=(256,))
    lights=[]
    num=1
    for num in range(file_nums):
        print("processing the",num+1,"'th image")
        filename=folderPath+file_list[num]
        pic=cv2.imread(filename=filename,flags=cv2.IMREAD_COLOR)

        #cv2.imshow(winname="minMap",mat=minMap)
        darkMap=dehaze3.DarkChannel(pic,15)


        for i in range(darkMap.shape[0]):
            for j in range(darkMap.shape[1]):
                lights.append(darkMap[i][j])
                result[darkMap[i][j]]+=1

    plt.hist(lights, bins=256, normed=True)
    plt.xlabel("Intensity")
    plt.ylabel("Probability")
    plt.show()



drawProb(folderPath="../data/H256x256/")