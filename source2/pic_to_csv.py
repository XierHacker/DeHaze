import numpy as np
import cv2
import os

def picToData(folderPath,kind):
    fileList = os.listdir(folderPath)
    num_of_samples = len(fileList)

    dataSet = np.zeros(shape=(num_of_samples, 256, 256, 3))
    labels = np.zeros(shape=(num_of_samples,))

    for i in range(num_of_samples):
        pic = cv2.imread(filename=folderPath + fileList[i], flags=cv2.IMREAD_COLOR)
        dataSet[i] = pic/255
        labels[i] = kind

    return dataSet, labels


x,y=picToData(folderPath="../data/H256x256/",kind=2)
print(x.shape)
print(y)
cv2.imshow(winname="pic", mat=x[0])
cv2.waitKey()
cv2.destroyAllWindows()









