import cv2
import numpy as np
pic=cv2.imread("image/H13.jpg")
print(pic.shape)

def getMostFrequent(img):
    A=np.zeros(shape=[1,3])
    g_array=np.zeros(shape=(256,))
    b_array=np.zeros(shape=(256,))
    r_array=np.zeros(shape=(256,))

    for row in range(img.shape[0]):
        for col in range(pic.shape[1]):
            g_array[pic[row,col,0]]+=1
            b_array[pic[row, col, 1]] += 1
            r_array[pic[row, col, 2]] += 1


    g_intensity=np.argmax(g_array)
    b_intensity=np.argmax(b_array)
    r_intensity=np.argmax(r_array)

    A[0,0]=g_intensity
    A[0,1]=b_intensity
    A[0,2]=r_intensity
    return A


A=getMostFrequent(pic)/255
print(A)