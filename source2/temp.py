import cv2
import numpy as np
import math
img=cv2.imread("../data/H256x256/H13.jpg")
print(img.shape)
A=np.zeros(shape=(1,3))
height=img.shape[0]
width=img.shape[1]
total_pixel=height*width
#print(total_pixel)
numpx = int(max(math.floor(total_pixel / 1000), 1))
#print(numpx)

g_img = img[:, :, 0]
b_img = img[:, :, 1]
r_img = img[:, :, 2]

g_img_vector=g_img.flatten()
g_img_vector.sort()
g_img_max=g_img_vector[-numpx:]
print(g_img_max.shape)
g_average=g_img_max.mean()
print(g_average)

b_img_vector=b_img.flatten()
b_img_vector.sort()
b_img_max=b_img_vector[-numpx:]
print(b_img_max.shape)
b_average=b_img_max.mean()
print(b_average)

r_img_vector=r_img.flatten()
r_img_vector.sort()
r_img_max=r_img_vector[-numpx:]
print(r_img_max.shape)
r_average=r_img_max.mean()
print(r_average)

#print(g_img)
#print(b_img)
#print(r_img.shape)

A[0,0]=g_average
A[0,1]=b_average
A[0,2]=r_average
print(A/255)
