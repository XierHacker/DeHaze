import numpy as np
import dehaze3
import dehaze
import metrics
import os
import math

def eval(folderPath):
    fileList=os.listdir(folderPath)
    for file in fileList:
        filename=folderPath+file
