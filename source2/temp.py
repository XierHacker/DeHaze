import cv2
import numpy as np
import dehaze3

img=cv2.imread("../data/QH256x256/QH1.jpg")
dehaze3.get_recover(img=img,size=9,kind=2)
#img2=cv2.equalizeHist(src=img[:,:,2])
#cv2.imshow("img2",img2)
#cv2.waitKey(0)
#cv2.destroyAllWindows()