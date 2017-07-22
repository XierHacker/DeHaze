import numpy as np
import matplotlib.pyplot as plt

x=np.linspace(start=-1,stop=1,num=100)
y=np.log(1-x)

plt.plot(x,y)
plt.show()




'''
import numpy as np
import cv2
import tensorflow as tf

pic=cv2.imread(filename="../data/H256x256/H1.jpg",flags=cv2.IMREAD_COLOR)
print(pic.shape)
dataSet=np.zeros(shape=(2,256,256,3))
dataSet[0]=pic/255
print(dataSet[0])

resized=tf.image.resize_image_with_crop_or_pad(image=dataSet[0],target_height=244,target_width=244)

print(resized)

with tf.Session() as sess:
    pic_resized=sess.run(resized)


cv2.imshow(winname="pic",mat=pic/255)
cv2.imshow(winname="dataSet",mat=dataSet[0])
cv2.imshow(winname="resized",mat=pic_resized)
cv2.waitKey()
cv2.destroyAllWindows()
'''

