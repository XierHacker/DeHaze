from alex_net import AlexNet
import tensorflow as tf
import numpy as np
import pic_to_csv

'''
train=np.random.normal(size=(100,224,224,3))
print(train.shape)

labels=np.random.random(size=100)*4
labels=labels.astype(np.int)
'''
x,labels=pic_to_csv.getDataSet()
print(x.shape)
print(labels.shape)

x=tf.image.resize_image_with_crop_or_pad(image=x,target_width=224,target_height=224)

with tf.Session() as sess:
    train=sess.run(x)

model=AlexNet()
model.fit(train,labels,epochs=10,batch_size=5)