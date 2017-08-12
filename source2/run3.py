import numpy as np
import cv2
import dehaze3
import metrics

#load img and show original image
pic=cv2.imread(filename="../data/H256x256/H64.jpg",flags=cv2.IMREAD_COLOR)
cv2.imshow(winname="original",mat=pic)


#trans to float
I = pic.astype('float64')/255

#get darkMap
darkMap=dehaze3.DarkChannel(I,15)
#print ("darkMap:",darkMap)
cv2.imshow(winname="darkmap",mat=darkMap)
#print ("darkMap.shape:",darkMap.shape)

#atmosphere light
A=dehaze3.AtmLight(I,darkMap,kind=2)
#print ("A:",A)
#print(A.shape)


#transMap
transMap_estimate = dehaze3.TransmissionEstimate(I,A,15)
cv2.imshow("TransMap_estimate:",transMap_estimate)
#print("shape of transMap_estimate:",transMap_estimate.shape)

#transMap_refine
transMap_refine=dehaze3.TransmissionRefine(pic,transMap_estimate)
cv2.imshow("TransMap_refine:",transMap_refine)
#print("shape of transMap_refine:",transMap_refine.shape)

#recover
recover=dehaze3.Recover(I,transMap_refine,A,0.1)
#print ("recover:",recover)
cv2.imshow("recover",recover)


recover2=(recover*255).astype(np.uint8)
#print(recover2)
cv2.imshow("recover2",recover2)

recover3=dehaze3.recoverEnhancement(recover,kind=2)
cv2.imshow("recover3",recover3)
print(metrics.get_all_metrics(I))
print(metrics.get_all_metrics(recover))
print(metrics.get_all_metrics(recover3))
cv2.waitKey(0)
cv2.destroyAllWindows()