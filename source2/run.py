import numpy as np
import cv2
import dehaze

#load img and show original image
pic=cv2.imread(filename="../data/12.jpg",flags=cv2.IMREAD_COLOR)
cv2.imshow(winname="original",mat=pic)

#trans to float
I = pic.astype('float64')/255

#get darkMap
darkMap=dehaze.DarkChannel(I,15)
print ("darkMap:",darkMap)
cv2.imshow(winname="darkmap",mat=darkMap)
print ("darkMap.shape:",darkMap.shape)

#atmosphere light
A=dehaze.AtmLight(I,darkMap)
print ("A:",A)
print(A.shape)


#transMap
transMap_estimate = dehaze.TransmissionEstimate(I,A,15)
cv2.imshow("TransMap_estimate:",transMap_estimate)
print("shape of transMap_estimate:",transMap_estimate.shape)

#transMap_refine
transMap_refine=dehaze.TransmissionRefine(pic,transMap_estimate)
cv2.imshow("TransMap_refine:",transMap_refine)
print("shape of transMap_refine:",transMap_refine.shape)

#recover
recover=dehaze.Recover(I,transMap_refine,A,0.1)
#print (recover)
cv2.imshow("recover",recover)

cv2.waitKey(0)
cv2.destroyAllWindows()