import numpy as np
import cv2
import dehaze2

#load img and show original image
pic=cv2.imread(filename="../data/H256x256/H18.jpg",flags=cv2.IMREAD_COLOR)
cv2.imshow(winname="original",mat=pic)

#trans to float
I = pic.astype('float64')/255

#get darkMap
darkMap=dehaze2.DarkChannel(I,15)
print ("darkMap:",darkMap)
cv2.imshow(winname="darkmap",mat=darkMap)
print ("darkMap.shape:",darkMap.shape)

#atmosphere light
A1,A2=dehaze2.AtmLight(I,darkMap)
print ("A1:",A1)
print(A1.shape)
print ("A2:",A2)
print(A2.shape)


#transMap
transMap_estimate1 = dehaze2.TransmissionEstimate(I,A1,15)
cv2.imshow("TransMap_estimate1:",transMap_estimate1)
print("shape of transMap_estimate1:",transMap_estimate1.shape)

transMap_estimate2 = dehaze2.TransmissionEstimate(I,A2,15)
cv2.imshow("TransMap_estimate2:",transMap_estimate2)
print("shape of transMap_estimate2:",transMap_estimate2.shape)

#transMap_refine
transMap_refine1=dehaze2.TransmissionRefine(pic,transMap_estimate1)
cv2.imshow("TransMap_refine1:",transMap_refine1)
print("shape of transMap_refine1:",transMap_refine1.shape)

transMap_refine2=dehaze2.TransmissionRefine(pic,transMap_estimate2)
cv2.imshow("TransMap_refine2:",transMap_refine2)
print("shape of transMap_refine2:",transMap_refine2.shape)

transMap=(transMap_refine1+transMap_refine2)/2
#recover
recover=dehaze2.Recover(I,transMap,(A1+A2)/2,0.1)
#print (recover)
cv2.imshow("recover",recover)

cv2.waitKey(0)
cv2.destroyAllWindows()