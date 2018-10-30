# Coder: Wenxin Xu
# Github: https://github.com/wenxinxu/resnet_in_tensorflow
# ==============================================================================
'''
This is the resnet structure
'''
import numpy as np
from hyper_parameters import *
import cifar10_input


BN_EPSILON = 0.001

def activation_summary(x):
    '''
    :param x: A Tensor
    :return: Add histogram summary and scalar summary of the sparsity of the tensor
    '''
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def create_variables(name, shape, initializer=tf.contrib.layers.xavier_initializer(), is_fc_layer=False):
    '''
    :param name: A string. The name of the new variable
    :param shape: A list of dimensions
    :param initializer: User Xavier as default.
    :param is_fc_layer: Want to create fc layer variable? May use different weight_decay for fc
    layers.
    :return: The created variable
    '''
    
    ## TODO: to allow different weight decay to fully connected layer and conv layer
    regularizer = tf.contrib.layers.l2_regularizer(scale=FLAGS.weight_decay)


    new_variables = tf.get_variable(name, shape=shape, initializer=initializer,
                                    regularizer=regularizer)
    # 该变量的正则化损失自动加到tf.GraphKeys.REGULARIZATION_LOSSES集合
    return new_variables


def output_layer(input_layer, num_labels):
    '''
    :param input_layer: 2D tensor
    :param num_labels: int. How many output labels in total? (10 for cifar10 and 100 for cifar100)
    :return: output layer Y = WX + B
    '''
    input_dim = input_layer.get_shape().as_list()[-1]
    fc_w = create_variables(name='fc_weights', shape=[input_dim, num_labels], is_fc_layer=True,
                            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    fc_b = create_variables(name='fc_bias', shape=[num_labels], initializer=tf.zeros_initializer())

    fc_h = tf.matmul(input_layer, fc_w) + fc_b
    return fc_h


def batch_normalization_layer(input_layer, dimension,is_traning):
    '''
    Helper function to do batch normalziation
    :param input_layer: 4D tensor
    :param dimension: input_layer.get_shape().as_list()[-1]. The depth of the 4D tensor
    :return: the 4D tensor after being normalized
    # '''
    # mean, variance = tf.nn.moments(input_layer, axes=[0, 1, 2])
    # beta = tf.get_variable('beta', dimension, tf.float32,
    #                            initializer=tf.constant_initializer(0.0, tf.float32))
    # gamma = tf.get_variable('gamma', dimension, tf.float32,
    #                             initializer=tf.constant_initializer(1.0, tf.float32))
    # bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, beta, gamma, BN_EPSILON)
    #
    # return bn_layer
    if is_traning:
        reuse=False
    else:
        reuse=True
    return tf.layers.batch_normalization(input_layer,epsilon=BN_EPSILON,center=True,scale=True,training=is_traning,reuse=tf.AUTO_REUSE)


def conv_bn_relu_layer(input_layer, filter_shape, stride,is_traning):
    '''
    A helper function to conv, batch normalize and relu the input tensor sequentially
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :return: 4D tensor. Y = Relu(batch_normalize(conv(X)))
    '''

    out_channel = filter_shape[-1]
    filter = create_variables(name='conv', shape=filter_shape)

    conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')

    bn_layer = batch_normalization_layer(conv_layer, out_channel,is_traning=is_traning)

    output = tf.nn.relu(bn_layer)
    return output


def bn_relu_conv_layer(input_layer, filter_shape, stride,is_traning):
    '''
    A helper function to batch normalize, relu and conv the input layer sequentially
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :return: 4D tensor. Y = conv(Relu(batch_normalize(X)))
    '''

    in_channel = input_layer.get_shape().as_list()[-1]

    bn_layer = batch_normalization_layer(input_layer, in_channel,is_traning=is_traning)
    relu_layer = tf.nn.relu(bn_layer)

    filter = create_variables(name='conv', shape=filter_shape)
    conv_layer = tf.nn.conv2d(relu_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    return conv_layer



def residual_block(input_layer, output_channel, is_traning,first_block=False):
    '''
    Defines a residual block in ResNet
    :param input_layer: 4D tensor
    :param output_channel: int. return_tensor.get_shape().as_list()[-1] = output_channel
    :param first_block: if this is the first residual block of the whole network
    :return: 4D tensor.
    '''
    input_channel = input_layer.get_shape().as_list()[-1]

    # When it's time to "shrink" the image size, we use stride = 2
    if input_channel * 2 == output_channel:
        increase_dim = True
        stride = 2
    elif input_channel == output_channel:
        increase_dim = False
        stride = 1
    else:
        raise ValueError('Output and input channel does not match in residual blocks!!!')

    # The first conv layer of the first residual block does not need to be normalized and relu-ed.
    with tf.variable_scope('conv1_in_block'):
        if first_block:
            filter = create_variables(name='conv', shape=[3, 3, input_channel, output_channel])
            conv1 = tf.nn.conv2d(input_layer, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
        else:
            conv1 = bn_relu_conv_layer(input_layer, [3, 3, input_channel, output_channel], stride,is_traning=is_traning)

    with tf.variable_scope('conv2_in_block'):
        conv2 = bn_relu_conv_layer(conv1,[3, 3, output_channel, output_channel], 1,is_traning=is_traning)

    # When the channels of input layer and conv2 does not match, we add zero pads to increase the
    #  depth of input layers
    if increase_dim is True:
        pooled_input = tf.nn.avg_pool(input_layer, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='VALID')
        padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [input_channel // 2,
                                                                     input_channel // 2]])
    else:
        padded_input = input_layer

    output = conv2 + padded_input
    return output


def inference(input_tensor_batch, n, reuse, is_traning):
    '''
    The main function that defines the ResNet. total layers = 1 + 2n + 2n + 2n +1 = 6n + 2
    :param input_tensor_batch: 4D tensor
    :param n: num_residual_blocks
    :param reuse: To build train graph, reuse=False. To build validation graph and share weights
    with train graph, resue=True
    :return: last layer in the network. Not softmax-ed
    '''
    # [-1,w,h,3]
    layers = []  # c存储计算图中的各个张量tensor
    filter_depth = 64
    stage = [2, 3, 3, 2]
    # features_num=[1 2 4 8]
    with tf.variable_scope('conv0', reuse=reuse):
        conv0 = conv_bn_relu_layer(input_tensor_batch, [7, 7, 3, filter_depth], 1, is_traning=is_traning)
        conv0 = tf.nn.max_pool(conv0, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME')
        activation_summary(conv0)
        layers.append(conv0)
    # [-1,w/2,h/2,features=16]
    for i in range(stage[0]):
        with tf.variable_scope('conv1_%d' % i, reuse=reuse):
            if i == 0:
                conv1 = residual_block(layers[-1], filter_depth, first_block=True, is_traning=is_traning)
            else:
                conv1 = residual_block(layers[-1], filter_depth, is_traning=is_traning)
            activation_summary(conv1)
            layers.append(conv1)
    # [-1,w/2,h/2,features]
    for i in range(stage[1]):
        with tf.variable_scope('conv2_%d' % i, reuse=reuse):
            conv2 = residual_block(layers[-1], filter_depth * 2, is_traning=is_traning)
            activation_summary(conv2)
            layers.append(conv2)
    # [-1,w/4,h/4,features*2]
    for i in range(stage[2]):
        with tf.variable_scope('conv3_%d' % i, reuse=reuse):
            conv3 = residual_block(layers[-1], filter_depth * 4, is_traning=is_traning)
            layers.append(conv3)
        # assert conv3.get_shape().as_list()[1:] == [cifar10_input.IMG_HEIGHT/8, cifar10_input.IMG_WIDTH/8, filter_depth*4]
    # [-1,w/8,h/8,features*4]

    for i in range(stage[3]):
        with tf.variable_scope('conv4_%d' % i, reuse=reuse):
            conv4 = residual_block(layers[-1], filter_depth * 8, is_traning=is_traning)
            layers.append(conv4)
        assert conv4.get_shape().as_list()[1:] == [cifar10_input.IMG_HEIGHT / 16, cifar10_input.IMG_WIDTH / 16,
                                                   filter_depth * 8]
    # [-1,w/16,h/16,features*8]

    # four stage 每个stage里面有n个stage

    with tf.variable_scope('fc', reuse=reuse):
        in_channel = layers[-1].get_shape().as_list()[-1]
        bn_layer = batch_normalization_layer(layers[-1], in_channel, is_traning=is_traning)
        relu_layer = tf.nn.relu(bn_layer)  # [-1,w/4,h/4,64]
        # global_pool = tf.reduce_mean(relu_layer, [1, 2])  #  求平均值 后为[-1,features*4]
        # Use avg_pool as an alternative since OpenCV does not support reduce_mean
        global_pool = tf.nn.avg_pool(relu_layer, [1, cifar10_input.IMG_HEIGHT / 16, cifar10_input.IMG_WIDTH / 16, 1],
                                     [1, cifar10_input.IMG_HEIGHT / 16, cifar10_input.IMG_WIDTH / 16, 1],
                                     padding='VALID')
        global_pool = tf.contrib.layers.flatten(global_pool, scope='faltten')

        assert global_pool.get_shape().as_list()[-1:] == [filter_depth * 8]
        output1 = output_layer(global_pool, cifar10_input.NUM_CLASS)  # fc [-1 ,10] 10为类别
        output = tf.nn.sigmoid(output1)
        layers.append(output)

    return layers[-1]


def inference_return_conv(input_tensor_batch, n, reuse,is_traning):
    '''
    The main function that defines the ResNet. total layers = 1 + 2n + 2n + 2n +1 = 6n + 2
    :param input_tensor_batch: 4D tensor
    :param n: num_residual_blocks
    :param reuse: To build train graph, reuse=False. To build validation graph and share weights
    with train graph, resue=True
    :return: last layer in the network. Not softmax-ed
    '''
    #[-1,w,h,3]
    layers = [] #c存储计算图中的各个张量tensor
    filter_depth=64
    stage=[2, 3, 3, 2]
    # features_num=[1 2 4 8]
    with tf.variable_scope('conv0', reuse=reuse):
        conv0 = conv_bn_relu_layer(input_tensor_batch, [7, 7, 3, filter_depth], 1,is_traning=is_traning)
        conv0=tf.nn.max_pool(conv0,[1,3,3,1],[1,2,2,1],padding='SAME')
        activation_summary(conv0)
        layers.append(conv0)
    # [-1,w/2,h/2,features=16]
    for i in range(stage[0]):
        with tf.variable_scope('conv1_%d' %i, reuse=reuse):
            if i == 0:
                conv1 = residual_block(layers[-1], filter_depth, first_block=True,is_traning=is_traning)
            else:
                conv1 = residual_block(layers[-1], filter_depth,is_traning=is_traning)
            activation_summary(conv1)
            layers.append(conv1)
    # [-1,w/2,h/2,features]
    for i in range(stage[1]):
        with tf.variable_scope('conv2_%d' %i, reuse=reuse):
            conv2 = residual_block(layers[-1], filter_depth*2,is_traning=is_traning)
            activation_summary(conv2)
            layers.append(conv2)
    # [-1,w/4,h/4,features*2]
    for i in range(stage[2]):
        with tf.variable_scope('conv3_%d' %i, reuse=reuse):
            conv3 = residual_block(layers[-1], filter_depth*4,is_traning=is_traning)
            layers.append(conv3)
        # assert conv3.get_shape().as_list()[1:] == [cifar10_input.IMG_HEIGHT/8, cifar10_input.IMG_WIDTH/8, filter_depth*4]
    # [-1,w/8,h/8,features*4]

    for i in range(stage[3]):
        with tf.variable_scope('conv4_%d' %i, reuse=reuse):
            conv4 = residual_block(layers[-1], filter_depth*8,is_traning=is_traning)
            layers.append(conv4)
        assert conv4.get_shape().as_list()[1:] == [cifar10_input.IMG_HEIGHT/16, cifar10_input.IMG_WIDTH/16, filter_depth*8]
    # [-1,w/16,h/16,features*8]

    # four stage 每个stage里面有n个stage

    with tf.variable_scope('fc', reuse=reuse):
        in_channel = layers[-1].get_shape().as_list()[-1]
        bn_layer = batch_normalization_layer(layers[-1], in_channel,is_traning=is_traning)
        relu_layer = tf.nn.relu(bn_layer)  # [-1,w/4,h/4,64]
        # global_pool = tf.reduce_mean(relu_layer, [1, 2])  #  求平均值 后为[-1,features*4]
        # Use avg_pool as an alternative since OpenCV does not support reduce_mean
        global_pool=tf.nn.avg_pool(relu_layer,[1,cifar10_input.IMG_HEIGHT/16,cifar10_input.IMG_WIDTH/16,1],[1,cifar10_input.IMG_HEIGHT/16,cifar10_input.IMG_WIDTH/16,1],padding='VALID')
        global_pool=tf.contrib.layers.flatten(global_pool,scope='faltten')

        


        assert global_pool.get_shape().as_list()[-1:] == [filter_depth*8]
        output1 = output_layer(global_pool, cifar10_input.NUM_CLASS)  #fc [-1 ,10] 10为类别
        output = tf.nn.sigmoid(output1)
        layers.append(output)

    return layers[-1],conv0,conv1,conv2,conv3,conv4


def inference_bak(input_tensor_batch, n, reuse):
    '''
    The main function that defines the ResNet. total layers = 1 + 2n + 2n + 2n +1 = 6n + 2
    :param input_tensor_batch: 4D tensor
    :param n: num_residual_blocks
    :param reuse: To build train graph, reuse=False. To build validation graph and share weights
    with train graph, resue=True
    :return: last layer in the network. Not softmax-ed
    '''
    #[-1,w,h,3]
    layers = [] #c存储计算图中的各个张量tensor
    filter_depth=64
    with tf.variable_scope('conv0', reuse=reuse):
        conv0 = conv_bn_relu_layer(input_tensor_batch, [3, 3, 3, filter_depth], 1)
        conv0=tf.nn.max_pool(conv0,[1,3,3,1],[1,2,2,1],padding='SAME')
        activation_summary(conv0)
        layers.append(conv0)
    # [-1,w/2,h/2,features=16]
    for i in range(n):
        with tf.variable_scope('conv1_%d' %i, reuse=reuse):
            if i == 0:
                conv1 = residual_block(layers[-1], filter_depth, first_block=True)
            else:
                conv1 = residual_block(layers[-1], filter_depth)
            activation_summary(conv1)
            layers.append(conv1)
    # [-1,w/2,h/2,features]
    for i in range(n):
        with tf.variable_scope('conv2_%d' %i, reuse=reuse):
            conv2 = residual_block(layers[-1], filter_depth*2)
            activation_summary(conv2)
            layers.append(conv2)
    # [-1,w/4,h/4,features*2]
    for i in range(n):
        with tf.variable_scope('conv3_%d' %i, reuse=reuse):
            conv3 = residual_block(layers[-1], filter_depth*4)
            layers.append(conv3)
        assert conv3.get_shape().as_list()[1:] == [cifar10_input.IMG_HEIGHT/8, cifar10_input.IMG_WIDTH/8, filter_depth*4]
    # [-1,w/8,h/8,features*4]

    # three stage 每个stage里面有n个stage

    with tf.variable_scope('fc', reuse=reuse):
        in_channel = layers[-1].get_shape().as_list()[-1]
        bn_layer = batch_normalization_layer(layers[-1], in_channel)
        relu_layer = tf.nn.relu(bn_layer)  # [-1,w/4,h/4,64]
        global_pool = tf.reduce_mean(relu_layer, [1, 2])  #  求平均值 后为[-1,features*4]
        
        assert global_pool.get_shape().as_list()[-1:] == [filter_depth*4]
        output = output_layer(global_pool, cifar10_input.NUM_CLASS)  #fc [-1 ,10] 10为类别
        layers.append(output)

    return layers[-1]


def test_graph(train_dir='logs'):
    '''
    Run this function to look at the graph structure on tensorboard. A fast way!
    :param train_dir:
    '''
    input_tensor = tf.constant(np.ones([128, 32, 32, 3]), dtype=tf.float32)
    result = inference(input_tensor, 2, reuse=False)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    summary_writer = tf.train.SummaryWriter(train_dir, sess.graph)
