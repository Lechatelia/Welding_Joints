import cv2
import tensorflow as tf
import numpy as np
import random

y = tf.constant([1,2,3,4,5,6], name='y',dtype=tf.float32)
y_ = tf.constant([0,1,2,3,4,5], name='Y_',dtype=tf.float32)
y = tf.reshape(y,[2,3])
y_ = tf.reshape(y_,[2,3])
z= tf.constant([1,2], name='z',dtype=tf.float32)
z=tf.reshape(z,[2,-1])


result=[]
result.append(tf.subtract(y,y_))
result.append(tf.multiply(y,y_))
result.append(tf.multiply(y,z))
with tf.Session() as sess:
    result=sess.run(result)
    for i in result:

        print(i)
    # result=sess.run(multiply)
    # print(result)
# y = tf.constant(0.5, shape=[7],name='y',dtype=tf.float32)
# y_ = tf.constant([0.6, 0.3,0.4,0.6,0.6,0.5,0.8], name='Y_',dtype=tf.float32)
# y_ = tf.constant([[9, 8], [7, 6], [10, 11]], name='x')
# b = tf.constant(1, name='b')

# a = tf.Variable(tf.zeros([3,3]))
# result=tf.zeros(y.get_shape().as_list()[0])

# result = tf.where(tf.greater(tf.abs((y-y_),"abs"),tf.constant(0.15,shape=y.get_shape(),dtype=tf.float32)),tf.constant(0,shape=y.get_shape(),dtype=tf.float32),tf.constant(1,shape=y.get_shape(),dtype=tf.float32))
# y=23
# y_=24
# # result = tf.where(tf.greater(y,y_),tf.abs(y-y_)*10,tf.abs(y-y_))
# result = tf.where(tf.greater(y,y_),y,y_)
# z = tf.where(tf.greater(y,y_),y_,y)
# z1=tf.to_int32(z)
# z2=tf.to_int32(result)
# #
#
# # result_mean=tf.reduce_mean(result)
# # Create a session to compute
# with tf.Session() as sess:
#     result=sess.run(result)
#     z=sess.run(z)
#     print(result)
#     # print(sess.run(result_mean))
#     print(z)

# img = cv2.imread("test.jpg")
#
# # img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = np.float32(img)/255
# cv2.imshow("Image",img)
# mask = np.zeros((50,50,1),dtype=np.float32)
# for i in range(20):
#     for j in range(20):
#         mask[i][j]=-0.5
#         mask[i+30][j+30]=0.5
#
# mask = cv2.resize(mask,(658,832))
#
# mask=cv2.cvtColor(mask,cv2.COLOR_GRAY2RGB)
# cv2.imshow("a",mask)
# cv2.addWeighted(img,0.5,mask,0.5,0,mask)
# cv2.imshow('hunh',mask)
# cv2.waitKey(0)


# cv2.destroyAllWindows()
# for i in  range(10):
#     print(random.randint(0, 1))
#
# a=[[[i*j*k for i in range(0,3)]for j in range(0,3)] for k in range(0,3)]
# # b=[[j*i for i in range(0,3)]for j in range(0,3)]
# print(a)
# # print(b)
# a=np.array(a)
# # b=np.array(b)
# print((list(a.shape)))
# # print(a+b);
# for n in a:
#     print(n)
# np.random.shuffle(a)
#
# print(len(a))







#
# print(random.randint(0, 2))
# print(random.randint(0, 2))
# print(random.randint(0, 2))
# print(random.randint(0, 2))
# print(random.randint(0, 2))


# c=[i for i in range(7)]
# print(c[-2:])

r1 = np.array([1.2, 2, 3, 4],dtype=np.float32)
r2 = np.array([1.1, 1.8, 3.3, 4.4],dtype=np.float32)
cha = r1 - r2
print(cha)
error = np.mean(np.abs(cha), axis=0)
print(error)