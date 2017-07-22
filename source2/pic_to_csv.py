import numpy as np
import matplotlib.image as mping
import os

def picToData(folderPath,kind):
    fileList = os.listdir(folderPath)
    num_of_samples = len(fileList)

    dataSet = np.zeros(shape=(num_of_samples, 256, 256, 3))
    labels = np.zeros(shape=(num_of_samples,))

    for i in range(num_of_samples):
        pic = mping.imread(folderPath + fileList[i])
        dataSet[i] = pic/255
        labels[i] = kind

    return dataSet, labels


def getDataSet():
    dataSet_QingJie, labels_Qingjie = picToData(folderPath="../data/QingJie256x256/", kind=1)
    dataSet_QH, labels_QH = picToData(folderPath="../data/QH256x256/", kind=2)
    dataSet_H, labels_H = picToData(folderPath="../data/H256x256/", kind=3)

    print(dataSet_QingJie.shape, dataSet_QH.shape, dataSet_H.shape)
    print(labels_Qingjie.shape, labels_QH.shape, labels_H.shape)

    total_num = dataSet_QingJie.shape[0] + dataSet_QH.shape[0] + dataSet_H.shape[0]
    dataSet_total = np.zeros(shape=(total_num, 256, 256, 3))
    labels_total = np.zeros(shape=(total_num,))

    dataSet_total[0:dataSet_QingJie.shape[0]] = dataSet_QingJie[:]
    labels_total[0:dataSet_QingJie.shape[0]] = labels_Qingjie[:]

    dataSet_total[dataSet_QingJie.shape[0]:dataSet_QingJie.shape[0] + dataSet_QH.shape[0]] = dataSet_QH[:]
    labels_total[dataSet_QingJie.shape[0]:dataSet_QingJie.shape[0] + dataSet_QH.shape[0]] = labels_QH[:]

    dataSet_total[dataSet_QingJie.shape[0] + dataSet_QH.shape[0]:] = dataSet_H[:]
    labels_total[dataSet_QingJie.shape[0] + dataSet_QH.shape[0]:] = labels_H[:]

    return dataSet_total,labels_total









