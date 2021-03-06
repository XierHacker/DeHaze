from __future__ import print_function,division
import numpy as np
import cv2
import dehaze_pro

pic=cv2.imread(filename="../data/H256x256/H26.jpg",flags=cv2.IMREAD_COLOR)
cv2.imshow(winname="original",mat=pic)
minMap=dehaze_pro.getMinMap(pic)
cv2.imshow(winname="minMap",mat=minMap)

pad,darkMap=dehaze_pro.getDarkMap(minMap)
print ("darkMap:",darkMap)
cv2.imshow(winname="pad",mat=pad)
cv2.imshow(winname="darkmap",mat=darkMap)
#print ("darkMap.shape:",darkMap.shape)
A=dehaze_pro.getAtmosphericLight(pic,darkMap)
#print ("A:",A)

#transMap
transMap=dehaze_pro.getTransmissionMap(darkMap,A)
cv2.imshow("TransMap",transMap)

recover=dehaze_pro.getRecoverMap(pic,transMap,A)
#print (recover)
cv2.imshow("recover",recover)
cv2.waitKey(0)
cv2.destroyAllWindows()