import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import dehaze

#dict to get pixel light info of dark map
result=np.zeros(shape=(256,))
accum=np.zeros(shape=(256,))
lights=[]
num=1
for num in range(1,177):
    print("processing the",num,"'th image")
    filename="../data/H256x256/H"+str(num)+".jpg"
    pic=cv2.imread(filename=filename,flags=cv2.IMREAD_COLOR)
    #cv2.imshow(winname="original",mat=pic)
    minMap=dehaze.getMinMap(pic)
    #cv2.imshow(winname="minMap",mat=minMap)
    pad,darkMap=dehaze.getDarkMap(minMap)
    #print ("darkMap:",darkMap)
    #cv2.imshow(winname="pad",mat=pad)
    #cv2.imshow(winname="darkmap",mat=darkMap)
    # print ("darkMap.shape:",darkMap.shape)
    #print(type(minMap))

    for i in range(darkMap.shape[0]):
        for j in range(darkMap.shape[1]):
            lights.append(darkMap[i][j])
            result[minMap[i][j]]+=1

'''
total=result.sum()
for l in range(256):
    accum[l]=result[:l+1].sum()/total

plt.plot(range(256),accum)
'''

plt.hist(lights,bins=256,normed=True)
plt.xlabel("Intensity")
plt.ylabel("Probability")
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()

