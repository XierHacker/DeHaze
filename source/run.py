from __future__ import print_function,division
import numpy as np
import cv2
import dehaze

pic=cv2.imread(filename="6.png",flags=cv2.IMREAD_COLOR)
cv2.imshow(winname="original",mat=pic)
minMap=dehaze.getMinMap(pic)
cv2.imshow(winname="minMap",mat=minMap)

pad,darkMap=dehaze.getDarkMap(minMap)
print ("darkMap:",darkMap)
cv2.imshow(winname="pad",mat=pad)
cv2.imshow(winname="darkmap",mat=darkMap)
#print ("darkMap.shape:",darkMap.shape)
A=dehaze.getAtmosphericLight(darkMap)
#print ("A:",A)

#transMap
transMap=dehaze.getTransmissionMap(darkMap,A)
cv2.imshow("TransMap",transMap)

recover=dehaze.getRecoverMap(pic,transMap,A)
#print (recover)
cv2.imshow("recover",recover)
cv2.waitKey(0)
cv2.destroyAllWindows()
