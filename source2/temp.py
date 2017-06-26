import cv2
import numpy as np
import math
import metrics
from skimage.measure import shannon_entropy




img=cv2.imread("../data/H256x256/H13.jpg")
img=img
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

print(g_img.mean())
print(g_img.var())
print(metrics.contrast(g_img))
print(metrics.entropy(g_img))
print(shannon_entropy(g_img,base=math.e))
