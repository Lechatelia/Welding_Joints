# Coder: Wenxin Xu
# Github: https://github.com/wenxinxu/resnet_in_tensorflow
# ==============================================================================

from resnet import *
from datetime import datetime
import time
from cifar10_input import *
import pandas as pd
import cv2
from hyper_parameters import *
import  my_dataset



class Train(object):
    '''
    This Object is responsible for all the training and validation process
    '''

    def __init__(self,in_size_h=32,in_size_w=32):
        # Set up all the placeholders
        self.placeholders()
        self.in_size_h = IMG_HEIGHT
        self.in_size_w = IMG_WIDTH


    def placeholders(self):
        '''
        There are five placeholders in total.
        image_placeholder and label_placeholder are for train images and labels
        vali_image_placeholder and vali_label_placeholder are for validation imgaes and labels
        lr_placeholder is for learning rate. Feed in learning rate each time of training
        implements learning rate decay easily
        '''
        self.image_placeholder = tf.placeholder(dtype=tf.float32,
                                                shape=[FLAGS.train_batch_size, IMG_HEIGHT,
                                                        IMG_WIDTH, IMG_DEPTH])
        self.label_placeholder = tf.placeholder(dtype=tf.float32, shape=[FLAGS.train_batch_size,1])

        self.vali_image_placeholder = tf.placeholder(dtype=tf.float32, shape=[FLAGS.validation_batch_size,
                                                                IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
        self.vali_label_placeholder = tf.placeholder(dtype=tf.float32, shape=[FLAGS.validation_batch_size,1])

        self.lr_placeholder = tf.placeholder(dtype=tf.float32, shape=[])




    def build_train_validation_graph(self):
        '''
        This function builds the train graph and validation graph at the same time.
        
        '''
        global_step = tf.Variable(0, trainable=False)
        validation_step = tf.Variable(0, trainable=False)

        # Logits of training data and valiation data come from the same graph. The inference of
        # validation data share all the weights with train data. This is implemented by passing
        # reuse=True to the variable scopes of train graph
        #reuse-=true 共用权重
        logits = inference(self.image_placeholder, FLAGS.num_residual_blocks, reuse=False,is_traning=True)
        vali_logits = inference(self.vali_image_placeholder, FLAGS.num_residual_blocks, reuse=True,is_traning=False)

        # The following codes calculate the train loss, which is consist of the
        # softmax cross entropy and the relularization loss
        regu_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)   #正则化损失
        loss = self.loss(logits, self.label_placeholder) #计算预测损失
        tf.summary.scalar('prediction_train_loss', loss)
        self.full_loss = tf.add_n([loss] + regu_losses)

        # predictions = tf.nn.softmax(logits)
        predictions = logits
        # self.train_top1_error = self.top_k_error(predictions, self.label_placeholder, 1)   #计算准确率

        self.train_top1_error = self.error(logits=logits,labels=self.label_placeholder)


        # Validation loss
        self.vali_loss = self.loss(vali_logits, self.vali_label_placeholder)
        # vali_predictions = tf.nn.softmax(vali_logits)
        vali_predictions = vali_logits
        self.vali_top1_error = self.error(vali_logits, self.vali_label_placeholder)
        # self.vali_top1_error = self.top_k_error(vali_predictions, self.vali_label_placeholder, 1)

        self.train_op, self.train_ema_op = self.train_operation(global_step, self.full_loss,
                                                                self.train_top1_error) #更改优化器
        self.val_op = self.validation_op(validation_step, self.vali_top1_error, self.vali_loss)




    def train(self):
        '''
          This is the main function for training
        '''

        # For the first step, we are loading all training images and validation images into the
        # memory
        all_data, all_labels = prepare_train_data(padding_size=FLAGS.padding_size)
        all_data = np.array(all_data).reshape(-1,1)
        all_labels = np.array(all_labels).reshape(-1,1) #作为标记只能是1维的(batch_size, )而不是(batch_size,1 )
        vali_data, vali_labels = read_validation_data()
        vali_data = np.array(vali_data).reshape(-1,1)
        vali_labels = np.array(vali_labels).reshape(-1,1)
        #存储数据集的路径和标记 ，第一维大小为总数据集的个数
        train_dataset=my_dataset.My_DataSet(all_data,all_labels)
        vali_dataset=my_dataset.My_DataSet(vali_data,vali_labels)

        # Build the graph for train and validation
        self.build_train_validation_graph()
        # Initialize a saver to save checkpoints. Merge all summaries, so we can run all
        # summarizing operations by running summary_op. Initialize a new session
        saver = tf.train.Saver(tf.global_variables(),max_to_keep=10)
        summary_op = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        sess = tf.Session()
        # If you want to load from a checkpoint
        if FLAGS.is_use_ckpt is True:
            saver.restore(sess, FLAGS.ckpt_path)
            print ('Restored from checkpoint...')
        else:
            sess.run(init)

        # This summary writer object helps write summaries on tensorboard
        summary_writer = tf.summary.FileWriter(train_dir, sess.graph)


        # These lists are used to save a csv file at last
        step_list = []
        train_error_list = []
        val_error_list = []

        print ('Start training...')
        print ('----------------------------')

        for step in range(FLAGS.train_steps):

            train_batch_data,train_batch_labels=train_dataset.next_batch(FLAGS.train_batch_size)
            #读取图片
            train_image_batch =self.generate__image_batch(train_batch_data,is_train=True)
            # 数据增强
            #train_image_batch=whitening_image(train_image_batch)
            random_shuffle_RGB(train_image_batch)
            random_crop_and_flip(train_image_batch,padding_size=FLAGS.padding_size)

            vali_batch_data,validation_batch_labels=vali_dataset.next_batch(FLAGS.validation_batch_size)
            vali_image_batch=self.generate__image_batch(vali_batch_data,is_train=False)

            #validation_batch_data, validation_batch_labels = self.generate_vali_batch(vali_data,
                                                           # vali_labels, FLAGS.validation_batch_size)

            # Want to validate once before training. You may check the theoretical validation
            # loss first
            if step % FLAGS.report_freq == 0:

                if FLAGS.is_full_validation is True:
                    validation_loss_value, validation_error_value = self.full_validation(loss=self.vali_loss,
                                            top1_error=self.vali_top1_error, vali_data=vali_data,
                                            vali_labels=vali_labels, session=sess,
                                            batch_data=train_batch_data, batch_label=train_batch_labels)

                    vali_summ = tf.Summary()
                    vali_summ.value.add(tag='full_validation_error',
                                        simple_value=validation_error_value.astype(np.float))
                    summary_writer.add_summary(vali_summ, step)
                    summary_writer.flush()

                else:
                    _, validation_error_value, validation_loss_value = sess.run([self.val_op,
                                                                     self.vali_top1_error,
                                                                 self.vali_loss],
                                                {self.image_placeholder: train_image_batch,
                                                 self.label_placeholder: train_batch_labels,
                                                 self.vali_image_placeholder: vali_image_batch,
                                                 self.vali_label_placeholder: validation_batch_labels,
                                                 self.lr_placeholder: FLAGS.init_lr})

                val_error_list.append(validation_error_value)


            start_time = time.time()

            _, _, train_loss_value, train_error_value = sess.run([self.train_op, self.train_ema_op,
                                                           self.full_loss, self.train_top1_error],
                                {self.image_placeholder: train_image_batch,
                                  self.label_placeholder: train_batch_labels,
                                  self.vali_image_placeholder: vali_image_batch,
                                  self.vali_label_placeholder: validation_batch_labels,
                                  self.lr_placeholder: FLAGS.init_lr})
            duration = time.time() - start_time


            if step % FLAGS.report_freq == 0:
                summary_str = sess.run(summary_op, {self.image_placeholder: train_image_batch,
                                                    self.label_placeholder: train_batch_labels,
                                                    self.vali_image_placeholder: vali_image_batch,
                                                    self.vali_label_placeholder: validation_batch_labels,
                                                    self.lr_placeholder: FLAGS.init_lr})
                summary_writer.add_summary(summary_str, step)

                num_examples_per_step = FLAGS.train_batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)

                format_str = ('%s: step %d, loss = %.4f (%.1f examples/sec; %.3f ' 'sec/batch)')
                print(format_str % (datetime.now(), step, train_loss_value, examples_per_sec,
                                    sec_per_batch))
                print('Train top1 error = ', train_error_value)
                print('Train loss = ', train_loss_value)
                print('Validation top1 error = %.4f' % validation_error_value)
                print('Validation loss = ', validation_loss_value)
                print('----------------------------')

                step_list.append(step)
                train_error_list.append(train_error_value)



            if step == FLAGS.decay_step0 or step == FLAGS.decay_step1:
                FLAGS.init_lr = 0.1 * FLAGS.init_lr
                print ('Learning rate decayed to ', FLAGS.init_lr)

            # Save checkpoints every 10000 steps
            if step % 100 == 0 or (step + 1) == FLAGS.train_steps:
                checkpoint_path = os.path.join(train_dir, 'model_ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

                df = pd.DataFrame(data={'step':step_list, 'train_error':train_error_list,
                                'validation_error': val_error_list})
                df.to_csv(train_dir + FLAGS.version + '_error.csv')  #将error写进csv


    def test(self, test_image_array):
        '''
        This function is used to evaluate the test data. Please finish pre-precessing in advance

        :param test_image_array: 4D numpy array with shape [num_test_images, img_height, img_width,
        img_depth]
        :return: the softmax probability with shape [num_test_images, num_labels]
        '''
        num_test_images = len(test_image_array)
        num_batches = num_test_images // FLAGS.test_batch_size
        remain_images = num_test_images % FLAGS.test_batch_size
        print ('%i test batches in total...' %num_batches)

        # Create the test image and labels placeholders
        self.test_image_placeholder = tf.placeholder(dtype=tf.float32, shape=[FLAGS.test_batch_size,
                                                        IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])

        # Build the test graph
        logits = inference(self.test_image_placeholder, FLAGS.num_residual_blocks, reuse=False)
        predictions = tf.nn.softmax(logits)

        # Initialize a new session and restore a checkpoint
        saver = tf.train.Saver(tf.all_variables())
        sess = tf.Session()

        saver.restore(sess, FLAGS.test_ckpt_path)
        print('Model restored from ', FLAGS.test_ckpt_path)

        prediction_array = np.array([]).reshape(-1, NUM_CLASS)
        # Test by batches
        for step in range(num_batches):
            if step % 10 == 0:
                print ('%i batches finished!' %step)
            offset = step * FLAGS.test_batch_size
            test_image_batch = test_image_array[offset:offset+FLAGS.test_batch_size, ...]

            batch_prediction_array = sess.run(predictions,
                                        feed_dict={self.test_image_placeholder: test_image_batch})

            prediction_array = np.concatenate((prediction_array, batch_prediction_array))

        # If test_batch_size is not a divisor of num_test_images
        if remain_images != 0:
            self.test_image_placeholder = tf.placeholder(dtype=tf.float32, shape=[remain_images,
                                                        IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH])
            # Build the test graph
            logits = inference(self.test_image_placeholder, FLAGS.num_residual_blocks, reuse=True)
            predictions = tf.nn.softmax(logits)

            test_image_batch = test_image_array[-remain_images:, ...]

            batch_prediction_array = sess.run(predictions, feed_dict={
                self.test_image_placeholder: test_image_batch})

            prediction_array = np.concatenate((prediction_array, batch_prediction_array))

        return prediction_array



    ## Helper functions
    def loss(self, logits, labels):
        '''
        Calculate the cross entropy loss given logits and true labels
        :param logits: 2D tensor with shape [batch_size, num_labels]
        :param labels: 1D tensor with shape [batch_size] 注意labels不是one-hot编码而直接是class的编号即index
        :return: loss tensor with shape [1]
        '''
        labels = tf.cast(labels, tf.float32)
        # loss_sum = tf.where(tf.greater(tf.abs((logits - labels), "abs"), 0.1), tf.abs(logits - labels),
        #                     tf.abs(logits - labels), name='loss_per_batch')
        # loss_sum = tf.where(tf.greater(tf.abs((logits - labels), "abs"), 0.1), tf.abs(logits - labels)*10-3/4,
        #                     tf.pow(tf.abs(logits - labels)*10,4)/4, name='loss_per_batch')

        # loss_sum=tf.nn.l1_loss(logits-labels,'l2-loss')
        loss_sum = tf.abs(logits - labels) * 10
        loss_mean = tf.reduce_mean(loss_sum, name="mean_loss_per_batch")
        return loss_mean


    def top_k_error(self, predictions, labels, k):
        '''
        Calculate the top-k error
        :param predictions: 2D tensor with shape [batch_size, num_labels]
        :param labels: 1D tensor with shape [batch_size, 1]
        :param k: int  每个样本的预测结果的前k个最大的数里面是否包含targets预测中的标签，一般都是取1，即取预测最大概率的索引与标签对比。
        :return: tensor with shape [1]
        '''
        batch_size = predictions.get_shape().as_list()[0]
        in_top1 = tf.to_float(tf.nn.in_top_k(predictions, labels, k=1))
        num_correct = tf.reduce_sum(in_top1)
        return (batch_size - num_correct) / float(batch_size)

    def error(self, logits, labels):
        '''
        Calculate the 回归 error
        :param predictions: 2D tensor with shape [batch_size, 1]
        :param labels: 1D tensor with shape [batch_size, 1]
        '''


        # labels = tf.cast(labels, tf.float32)

        error_sum = tf.where(tf.greater(tf.abs((logits - labels), "abs"), tf.constant(0.1,shape=logits.get_shape(),dtype=tf.float32)),
                            tf.constant(1, shape=logits.get_shape(), dtype=tf.float32),
                            tf.constant(0, shape=logits.get_shape(), dtype=tf.float32),
                            name='error_per_batch')

        error_mean = tf.reduce_mean(error_sum, name="mean_error_per_batch")
        return error_mean

        # error=tf.where(tf.greater(tf.abs((logits-labels), "abs_2"), 0.1),
        #                                    tf.constant(0, shape=labels.get_shape(), dtype=tf.float32),
        #                                    tf.constant(1, shape=labels.get_shape(), dtype=tf.float32),
        #                                    name='error_per_batch')
        # mean_error=tf.reduce_mean(error)
        #
        # return mean_error



    # ---------------------------- Image Reader --------------------------------
    def open_img_train(self, name, color='RGB'):
        """ Open an image
        Args:
            name	: Name of the sample
            color	: Color Mode (RGB/BGR/GRAY)
        """
        img = cv2.imread(os.path.join(full_data_dir, name))
        # print(name)
        # assert (img[0] > 0 & img[1] > 0)
        img=cv2.resize(img,(self.in_size_w,self.in_size_h))
        if color == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        elif color == 'BGR':
            return img
        elif color == 'GRAY':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return img
        else:
            print('Color mode supported: RGB/BGR. If you need another mode do it yourself :p')

    def open_img_valid(self, name, color='RGB'):
        """ Open an image
        Args:
            name	: Name of the sample
            color	: Color Mode (RGB/BGR/GRAY)
        """

        img = cv2.imread(os.path.join(vali_dir, name))
        # print(name)
        # assert (img[0]>0 & img[1]>0)
        img = cv2.resize(img, (self.in_size_h, self.in_size_w))
        if color == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        elif color == 'BGR':
            return img
        elif color == 'GRAY':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return img
        else:
            print('Color mode supported: RGB/BGR. If you need another mode do it yourself :p')


    def generate_vali_batch(self, vali_data, vali_label, vali_batch_size):
        '''
        If you want to use a random batch of validation data to validate instead of using the
        whole validation data, this function helps you generate that batch
        :param vali_data: 4D numpy array
        :param vali_label: 1D numpy array
        :param vali_batch_size: int
        :return: 4D numpy array and 1D numpy array
        '''
        vali_data_batch=[]
        offset = np.random.choice(valid_EPOCH_SIZE - vali_batch_size, 1)[0]
        vali_data_batch_dir = vali_data[offset:offset+vali_batch_size, ...]


        # 可在此进行图片增强处理 添加padding之类的
        # opencv读取3D numpy array并存放添加到数组中这样组成4D numpy array


        for dir in vali_data_batch_dir:
            vali_data_batch.append(self.open_img_valid(dir))

        # 图片读取之后再处理
        #vali_data_batch = whitening_image(vali_data_batch)

        vali_label_batch = vali_label[offset:offset+vali_batch_size]
        return vali_data_batch, vali_label_batch



    #产生一个batch的图像数据，
    def generate__image_batch(self,train_data,padding_size=0,is_train=True):
        if is_train:
            train_img = np.zeros((FLAGS.train_batch_size, self.in_size_h, self.in_size_w, 3), dtype=np.float32)
            i=0
            while i<FLAGS.train_batch_size:
                if FLAGS.is_data_augmentation is True:
                    train_img[i] = cifar10_input.random_image_rotate(self.open_img_train(train_data[i][0]) / 256)
                else:
                    train_img[i] = self.open_img_train(train_data[i][0]) / 256

                i=i+1


            # cv2.imshow('image', train_img[1])
            # cv2.waitKey( )
            # pad_width = ((0, 0), (padding_size, padding_size), (padding_size, padding_size), (0, 0))
            # batch_data = np.pad(train_img, pad_width=pad_width, mode='constant', constant_values=0)
            # batch_data = random_crop_and_flip(batch_data, padding_size=FLAGS.padding_size)
            # 在这里所以验证一下数据的大小和尺寸
            #train_img = whitening_image(train_img)
            # 可在此debug看一下图片效果
            # cv2.imshow('image',train_img[1])
            # cv2.waitKey()
            #print('train_image_batch shape:  '+str(train_img.shape))
            return train_img
        else:
            test_image = np.zeros((FLAGS.validation_batch_size, self.in_size_h, self.in_size_w, 3), dtype=np.float32)
            i = 0
            while i < FLAGS.validation_batch_size:
                test_image[i] = self.open_img_train(train_data[i][0])/256  #因为为float
                i = i + 1
            #print('test_image_batch shape:  ' + str(test_image.shape))
            return test_image





    def generate_augment_train_batch(self, train_data, train_labels, train_batch_size,padding_size=0):
        '''
        This function helps generate a batch of train data, and random crop, horizontally flip
        and whiten them at the same time
        :param train_data: 4D numpy array
        :param train_labels: 1D numpy array
        :param train_batch_size: int
        :return: augmented train batch data and labels. 4D numpy array and 1D numpy array
        '''
        batch_data=[]
        offset = np.random.choice(EPOCH_SIZE - train_batch_size, 1)[0]  #获得随机起始点，避免溢出
        batch_data_dir = train_data[offset:offset+train_batch_size, ...]  #获得数据集
        #可在此进行图片增强处理 添加padding之类的
        #opencv读取3D numpy array并存放添加到数组中这样组成4D numpy array
        for dir in batch_data_dir:
            batch_data.append(self.open_img_train(dir))
        pad_width = ((0, 0), (padding_size, padding_size), (padding_size, padding_size), (0, 0))
        batch_data = np.pad(batch_data, pad_width=pad_width, mode='constant', constant_values=0)

        batch_data = random_crop_and_flip(batch_data, padding_size=FLAGS.padding_size)
        #在这里所以验证一下数据的大小和尺寸
        batch_data = whitening_image(batch_data)
        #可在此debug看一下图片效果
        batch_label = train_labels[offset:offset+FLAGS.train_batch_size]

        return batch_data, batch_label



    def train_operation(self, global_step, total_loss, top1_error):
        '''
        Defines train operations
        :param global_step: tensor variable with shape [1]
        :param total_loss: tensor with shape [1]
        :param top1_error: tensor with shape [1]
        :return: two operations. Running train_op will do optimization once. Running train_ema_op
        will generate the moving average of train error and train loss for tensorboard
        '''
        # Add train_loss, current learning rate and train error into the tensorboard summary ops
        tf.summary.scalar('learning_rate', self.lr_placeholder)
        tf.summary.scalar('train_loss', total_loss)
        tf.summary.scalar('train_top1_error', top1_error)

        # The ema object help calculate the moving average of train loss and train error
        ema = tf.train.ExponentialMovingAverage(FLAGS.train_ema_decay, global_step)
        train_ema_op = ema.apply([total_loss, top1_error])
        tf.summary.scalar('train_top1_error_avg', ema.average(top1_error))
        tf.summary.scalar('train_loss_avg', ema.average(total_loss))

        # opt = tf.train.MomentumOptimizer(learning_rate=self.lr_placeholder, momentum=0.9)

        #opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        # opt = tf.train.AdamOptimizer(self.learning_rate, epsilon=1e-8)



        min_learning_rate = tf.constant(0.001, name='y', dtype=tf.float32)
        learning_rate = tf.train.exponential_decay(0.01, global_step, 100, 0.99, staircase=True)
        learning_rate = tf.where(tf.greater(learning_rate,min_learning_rate),learning_rate,min_learning_rate)
        opt = tf.train.GradientDescentOptimizer(learning_rate)
        tf.summary.scalar('learning_rate_1', learning_rate)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # train_op = optimizer.minimize(loss)
            train_op = opt.minimize(total_loss, global_step=global_step)
        return train_op, train_ema_op


    def validation_op(self, validation_step, top1_error, loss):
        '''
        Defines validation operations
        :param validation_step: tensor with shape [1]
        :param top1_error: tensor with shape [1]
        :param loss: tensor with shape [1]
        :return: validation operation
        '''

        # This ema object help calculate the moving average of validation loss and error

        # ema with decay = 0.0 won't average things at all. This returns the original error
        ema = tf.train.ExponentialMovingAverage(0.0, validation_step)
        ema2 = tf.train.ExponentialMovingAverage(0.95, validation_step)


        val_op = tf.group(validation_step.assign_add(1), ema.apply([top1_error, loss]),
                          ema2.apply([top1_error, loss]))
        top1_error_val = ema.average(top1_error)
        top1_error_avg = ema2.average(top1_error)
        loss_val = ema.average(loss)
        loss_val_avg = ema2.average(loss)

        # Summarize these values on tensorboard
        tf.summary.scalar('val_top1_error', top1_error_val)
        tf.summary.scalar('val_top1_error_avg', top1_error_avg)
        tf.summary.scalar('val_loss', loss_val)
        tf.summary.scalar('val_loss_avg', loss_val_avg)
        return val_op


    def full_validation(self, loss, top1_error, session, vali_data, vali_labels, batch_data,
                        batch_label):
        '''
        Runs validation on all the 10000 valdiation images
        :param loss: tensor with shape [1]
        :param top1_error: tensor with shape [1]
        :param session: the current tensorflow session
        :param vali_data: 4D numpy array
        :param vali_labels: 1D numpy array
        :param batch_data: 4D numpy array. training batch to feed dict and fetch the weights
        :param batch_label: 1D numpy array. training labels to feed the dict
        :return: float, float
        '''
        num_batches = 10000 // FLAGS.validation_batch_size
        order = np.random.choice(10000, num_batches * FLAGS.validation_batch_size)
        vali_data_subset = vali_data[order, ...]
        vali_labels_subset = vali_labels[order]

        loss_list = []
        error_list = []

        for step in range(num_batches):
            offset = step * FLAGS.validation_batch_size
            feed_dict = {self.image_placeholder: batch_data, self.label_placeholder: batch_label,
                self.vali_image_placeholder: vali_data_subset[offset:offset+FLAGS.validation_batch_size, ...],
                self.vali_label_placeholder: vali_labels_subset[offset:offset+FLAGS.validation_batch_size],
                self.lr_placeholder: FLAGS.init_lr}
            loss_value, top1_error_value = session.run([loss, top1_error], feed_dict=feed_dict)
            loss_list.append(loss_value)
            error_list.append(top1_error_value)

        return np.mean(loss_list), np.mean(error_list)

if __name__ == '__main__':
    #maybe_download_and_extract()
    # Initialize the Train object
    train = Train()
    # Start the training session
    train.train()




