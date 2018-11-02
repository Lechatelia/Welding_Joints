from resnet import *
from datetime import datetime
import time
from cifar10_input import *
import pandas as pd
import cv2
from hyper_parameters import *
import  my_dataset
import os
import time
import numpy as np
import matplotlib.pyplot as plt

#请注意换到\\是用到Windows下面的 ，在ubuntu底下需要作出相应的更改 有两处写到txt处的需要更改
#每次预测之前请删除原本的文件夹

class Test(object):
    def __init__(self,images_dir,in_size_h=IMG_HEIGHT,in_size_w=IMG_WIDTH):
        # Set up all the placeholders
        self.images_dir=images_dir
        self.in_size_h = in_size_h
        self.in_size_w = in_size_w
        self.filenames=[]
        self.image_label=[]

    def read_imges_filenames(self):
        for root, dirs, files in os.walk(self.images_dir):
            for file in files:
                if file.split('.')[-1]!='txt':
                    self.filenames.append(os.path.join(self.images_dir,file))
                    self.image_label.append(float(file.split('_')[0]) / 100)
        print(self.filenames)

        # cv2.imshow('1',cv2.imread(self.filenames[1]))
        # cv2.waitKey()
    def generate_image_array(self,image_name):
        test_img = np.zeros((len(image_name), self.in_size_h, self.in_size_w, 3), dtype=np.float32)
        i = 0
        while i < len(image_name):
            test_img[i]  = cv2.resize(cv2.cvtColor(cv2.imread(image_name[i]), cv2.COLOR_BGR2RGB), (self.in_size_w, self.in_size_h))/256
            i = i + 1
        return test_img

    def test_one_image_show(self, dir ):
        '''

        '''


        # Create the test image and labels placeholders
        self.test_image_placeholder = tf.placeholder(dtype=tf.float32, shape=[1,
                                                                              IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])

        # Build the test graph
        logits, attention_out = inference(self.test_image_placeholder, FLAGS.num_residual_blocks, reuse=False,is_traning=False,return_conv=True)
        predictions = logits
        saver = tf.train.Saver(tf.global_variables())
        sess = tf.Session()
        saver.restore(sess, FLAGS.test_ckpt_path)
        print('Model restored from ', FLAGS.test_ckpt_path)
        # Test by batches

        # test_image = self.generate_image_array(dir)
        test_image = np.zeros((1, self.in_size_h, self.in_size_w, 3), dtype=np.float32)
        test_image[0]= cv2.resize(cv2.cvtColor(cv2.imread(dir), cv2.COLOR_BGR2RGB),
                                 (self.in_size_w, self.in_size_h)) / 256.0
        batch_prediction, attention_out= sess.run([logits, attention_out],
                                          feed_dict={self.test_image_placeholder: test_image})

        attention_out = sess.run(tf.transpose(attention_out,[0,3,1,2]))
        attention_out = np.squeeze(np.array(attention_out),0)
        for i in range(attention_out.shape[0]):
            img = (attention_out[i]*255).astype(np.uint8)
            # cv2.imshow('channel{i}'.format(i=i),img)
            # cv2.waitKey(10)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            cv2.imwrite("/home/lechatelia/Desktop/Codes/welding_joints_log/attention_img/channel{i}.jpg".format(i=i),img)
        # for j in range(len(conv_re)):
        #     conv0_transpose = sess.run(tf.transpose(conv_re[j], [3, 0, 1, 2]))
        #     fig0, ax0 = plt.subplots(nrows=1, ncols=16, figsize=(16, 1))
        #     for i in range(16):
        #         ax0[i].imshow(conv0_transpose[i][0])  # tensor的切片[row, column]
        #     plt.title('Conv_%d_16' %j)
        #     plt.savefig('Conv_%d_16' %j)
        #     plt.show()
        # return prediction_array  # ,predict_end



    def test_predict(self, test_image_array,write_txt=True):
        '''
        This function is used to evaluate the test data. Please finish pre-precessing in advance

        :param test_image_array: 4D numpy array with shape [num_test_images, img_height, img_width,
        img_depth]
        :return: the softmax probability with shape [num_test_images, num_labels]
        '''
        if write_txt:
            out_file = self.write_predictions()  # 写txt记录预测结果
        num_test_images = len(test_image_array)
        num_batches = num_test_images // FLAGS.test_batch_size
        remain_images = num_test_images % FLAGS.test_batch_size
        print('%i test batches in total...' % (num_batches + 1))

        # Create the test image and labels placeholders
        self.test_image_placeholder = tf.placeholder(dtype=tf.float32, shape=[FLAGS.test_batch_size,
                                                                              IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])

        # Build the test graph
        logits = inference(self.test_image_placeholder, FLAGS.num_residual_blocks, reuse=False,is_traning=False)
        # predictions = tf.nn.softmax(logits)
        # prediction_class = tf.argmax(predictions, 1)
        predictions = logits
        # prediction_class = tf.argmax(predictions, 1)
        # Initialize a new session and restore a checkpoint
        # saver = tf.train.Saver(tf.all_variables())
        saver = tf.train.Saver(tf.global_variables())

        sess = tf.Session()

        saver.restore(sess, FLAGS.test_ckpt_path)
        print('Model restored from ', FLAGS.test_ckpt_path)

        prediction_array = np.array([]).reshape(-1, NUM_CLASS)
        predict_end = np.array([]).reshape(-1)
        # Test by batches
        time_start = time.time()
        for step in range(num_batches):
            if step % 10 == 0:
                print('%i batches finished!' % step)
            offset = step * FLAGS.test_batch_size
            test_image_batch = test_image_array[offset:offset + FLAGS.test_batch_size]
            # test_image_batch = test_image_array[offset:offset + FLAGS.test_batch_size, ...]
            test_image = self.generate_image_array(test_image_batch)

            batch_prediction_array = sess.run([logits],
                                              feed_dict={self.test_image_placeholder: test_image})[0]
            #注意返回的是一个数列

            if write_txt:
                # 请注意换到\\是用到Windows下面的 ，在ubuntu底下需要作出相应的更改
                i = 0
                while i < len(test_image_batch):
                    out_file.write(
                        test_image_batch[i].split('/')[-1] + '\t\t\t' + str(batch_prediction_array[i][0]*100) + '\n')
                    i = i + 1
            prediction_array = np.concatenate((prediction_array, batch_prediction_array))
            # predict_end = np.concatenate((predict_end, class_num))

        # If test_batch_size is not a divisor of num_test_images
        if remain_images != 0:
            self.test_image_placeholder = tf.placeholder(dtype=tf.float32, shape=[remain_images,
                                                                                  IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
            # Build the test graph
            logits = inference(self.test_image_placeholder, FLAGS.num_residual_blocks, reuse=True,is_traning=False)
            predictions = logits
            # predictions = tf.nn.softmax(logits)
            # prediction_class = tf.argmax(predictions, 1)

            test_image_batch = test_image_array[-remain_images:, ...]
            test_image = self.generate_image_array(test_image_batch)
            batch_prediction_array = sess.run([logits], feed_dict={
                self.test_image_placeholder: test_image})[0]
            if write_txt:
                i = 0
                while i < len(test_image_batch):
                    # 请注意换到\\是用到Windows下面的 ，在ubuntu底下需要作出相应的更改
                    out_file.write(
                        test_image_batch[i].split('/')[-1] + '\t\t\t' + str(batch_prediction_array[i][0]*100) + '\n')
                    i = i + 1

            prediction_array = np.concatenate((prediction_array, batch_prediction_array))
            # predict_end = np.concatenate((predict_end, class_num))
        time_duration = time.time() - time_start
        time_per_image = time_duration / num_test_images
        print("总耗时：\t %.4f,总数量：\t%i,平均耗时：\t%.4f\n" % (time_duration, num_test_images, time_per_image))
        return prediction_array  # ,predict_end

    def write_predictions(self):
        relative_dir = '/predictions/'
        path=self.images_dir+relative_dir
        isExists = os.path.exists(path)
        if not isExists:
            # 如果不存在则创建目录
            # 创建目录操作函数
            os.makedirs(path)
        out_file = open(path + 'predictions.txt', 'w')
        return out_file

    def compound(self, imgdir, maskdir):
        img = cv2.imread(imgdir)
        img = cv2.resize(img, (IMG_WIDTH,IMG_HEIGHT))
        cv2.imwrite("originimage.jpg",img)
        cv2.imshow("origin",img)
        cv2.waitKey(10)
        mask = cv2.imread(maskdir)
        mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
        mask = cv2.bitwise_not(mask)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.imwrite("attention.jpg", mask)
        # mask = cv2.threshold(mask,110,255,cv2.CV_THRESH_BINARY)
        cv2.imshow("a",mask)
        cv2.waitKey(10)
        cv2.addWeighted(img,0.2,mask,0.8,0,mask)
        cv2.imwrite("attentionresult.jpg",mask)
        cv2.imshow('hunh',mask)
        cv2.waitKey(0)




if __name__ == '__main__':
    test=Test(cifar10_input.test_images_dir,IMG_HEIGHT,IMG_WIDTH)
    Mode = 1
    if Mode==0:
        test.read_imges_filenames()
        # prediction_array,predict_end=test.test_predict(np.array(test.filenames),write_txt=True)
        prediction_array=test.test_predict(np.array(test.filenames),write_txt=True)
        labels=np.array(test.image_label)
        error = np.mean(np.abs(np.squeeze(prediction_array,-1)-labels),axis=0)
        print(prediction_array)
        print("pridict error is {error}".format(error=error))
    # print(predict_end)
    elif Mode == 1:
        test.test_one_image_show('1234.jpg')
    elif Mode == 2:
        test.compound("1234.jpg","channel124.jpg")
