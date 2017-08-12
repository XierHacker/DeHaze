import numpy as np
import dehaze3
import dehaze
import metrics
import os
import math
import cv2
import matplotlib.pyplot as plt

def eval(folderPath):
    fileList=os.listdir(folderPath)
    num_of_files=len(fileList)
    result_img=np.zeros(shape=(num_of_files,3))
    result_dehaze=np.zeros(shape=(num_of_files,3))
    result_dehaze3=np.zeros(shape=(num_of_files,3))
    index=0
    for file in fileList:
        print("processing ",index+1,"\'th image")
        filename=folderPath+file
        img = cv2.imread(filename)
        recover_dehaze=dehaze.get_recover(img,9)
        recover_dehaze3=dehaze3.get_recover(img,9,kind=1)
        #result is a 3 element list
        metrics_img=metrics.get_all_metrics(img/255)
        metrics_dehaze=metrics.get_all_metrics(recover_dehaze)
        metrics_dehaze3 = metrics.get_all_metrics(recover_dehaze3)

        #
        result_img[index,:]=metrics_img[:]
        result_dehaze[index,:]=metrics_dehaze[:]
        result_dehaze3[index,:]=metrics_dehaze3[:]
        index+=1
    return result_img,result_dehaze,result_dehaze3

def draw(result_img,result_dehaze,result_dehaze3):
    num_of_samples=result_img.shape[0]
    #draw contrast
    fig1=plt.figure(1)
    plt.title("Contrast In Different Method")
    plt.plot(range(num_of_samples),result_img[:,0],"r",
             range(num_of_samples),result_dehaze[:,0],"g",
             range(num_of_samples), result_dehaze3[:, 0],"b")
    plt.xlabel("The Sample Number")
    plt.ylabel("Contrast")

    # draw entropy
    fig1 = plt.figure(2)
    plt.title("Entropy In Different Method")
    plt.plot(range(num_of_samples), result_img[:, 1],"r",
             range(num_of_samples), result_dehaze[:, 1],"g",
             range(num_of_samples), result_dehaze3[:, 1],"b")
    plt.xlabel("The Sample Number")
    plt.ylabel("Entropy")

    # draw ave gradient
    fig1 = plt.figure(3)
    plt.title("Average Gradient In Different Method")
    plt.plot(range(num_of_samples), result_img[:, 2],"r",
             range(num_of_samples), result_dehaze[:, 2],"g",
             range(num_of_samples), result_dehaze3[:, 2],"b")
    plt.xlabel("The Sample Number")
    plt.ylabel("Average Gradient")

    plt.show()

def get_sheet(result_img,result_dehaze,result_dehaze3):
    num=result_img.shape[0]
    contrast_ave = np.zeros(shape=(3,))
    entropy_ave = np.zeros(shape=(3,))
    aveGradient_ave = np.zeros(shape=(3,))

    contrast_best=np.zeros(shape=(3,))
    entropy_best = np.zeros(shape=(3,))
    aveGradient_best = np.zeros(shape=(3,))
    for i in range(num):
        contrast_ave=np.array([result_img[i,0],result_dehaze[i,0],result_dehaze3[i,0]])
        entropy_ave=np.array([result_img[i,1],result_dehaze[i,1],result_dehaze3[i,1]])
        aveGradient_ave=np.array([result_img[i,2],result_dehaze[i,2],result_dehaze3[i,2]])

        contrast_best[np.argmax(contrast_ave)]+=1
        entropy_best[np.argmax(entropy_ave)]+=1
        aveGradient_best[np.argmax(aveGradient_ave)]+=1

    return contrast_best,entropy_best,aveGradient_best



result_img,result_dehaze,result_dehaze3=eval(folderPath="../data/QingJie256x256/")
print("result_img:")
print(result_img.shape)
print(result_img)
print()
print("result_dehaze:")
print(result_dehaze.shape)
print(result_dehaze)
print()
print("result_dehaze3")
print(result_dehaze3.shape)
print(result_dehaze3)

contrast_best,entropy_best,aveGradient_best=get_sheet(result_img,result_dehaze,result_dehaze3)
print(contrast_best)
print(entropy_best)
print(aveGradient_best)

draw(result_img,result_dehaze,result_dehaze3)