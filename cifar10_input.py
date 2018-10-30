# Coder: Wenxin Xu
# Github: https://github.com/wenxinxu/resnet_in_tensorflow
# ==============================================================================
import tarfile
from six.moves import urllib
import sys
import numpy as np
from hyper_parameters import *
import pickle
import os
from scipy import ndimage
import cv2
import numpy as np
import random


data_dir = 'cifar10_data'
#full_data_dir = 'cifar10_data/cifar-10-batches-py/data_batch_'
#vali_dir = 'cifar10_data/cifar-10-batches-py/test_batch'
DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

full_data_dir = '/home/lechatelia/Desktop/Doc/images/'  #存放训练数据集的txt路径只有数字不同
train_txt_name='train.txt'
vali_dir='/home/lechatelia/Desktop/Doc/images/' #存放验证数据集的txt路径
vali_txt_name='valid.txt'
NUM_TRAIN_BATCH = 1  # 数据集txt个数

test_images_dir = '/home/lechatelia/Desktop/Doc/images/train_images/'

IMG_HEIGHT = 448
IMG_WIDTH = 336
IMG_DEPTH = 3
NUM_CLASS = 1

TRAIN_RANDOM_LABEL = False # Want to use random label for train data?
VALI_RANDOM_LABEL = False # Want to use random label for validation?


EPOCH_SIZE = 10000 * NUM_TRAIN_BATCH  #图片数目
valid_EPOCH_SIZE=200;


def maybe_download_and_extract():
    '''
    Will download and extract the cifar10 data automatically
    :return: nothing
    '''
    dest_directory = data_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, float(count * block_size)
                                                             / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def _read_one_batch(path, is_random_label):
    '''
    The training data contains five data batches in total. The validation data has only one
    batch. This function takes the directory of one batch of data and returns the images and
    corresponding labels as numpy arrays

    :param path: the directory of one batch of data
    :param is_random_label: do you want to use random labels?
    :return: image numpy arrays and label numpy arrays
    '''
    fo = open(path, 'rb')
    dicts = pickle.load(fo)
    fo.close()

    data = dicts['data']
    if is_random_label is False:
        label = np.array(dicts['labels'])
    else:
        labels = np.random.randint(low=0, high=10, size=10000)
        label = np.array(labels)
    return data, label

def My_read_in_all_images( is_random_label = True,is_train_image=True):
    """
    This function reads all training or validation data, shuffles them if needed, and returns the
    images and the corresponding labels as numpy arrays

    :param address_list: a list of paths of cPickle files
    :param is_train_image: 是否用于train的dataset读取
    :return: concatenated numpy array of data and labels. Data are in 4D arrays: [num_images,
    image_height, image_width, image_depth] and labels are in 1D arrays: [num_images]
    """
    train_table = []
    train_image_dir = []
    train_image_label = []
    if is_train_image:
        dir=full_data_dir
        txt_name=train_txt_name
    else:
        dir=vali_dir
        txt_name=vali_txt_name
    for i in range(0, NUM_TRAIN_BATCH ):
        with open(dir +txt_name, 'r') as file_to_read:  #路径加txt的文件名
            while True:
                lines = file_to_read.readline()  # 整行读取数据
                if not lines:
                    break
                train_table.append(lines)
    if is_random_label:
        np.random.shuffle(train_table)
    for info in train_table:
        train_image_dir.append(info.split(' ')[0])
        train_image_label.append(float(info.split(' ')[-1]) / 100)
        #二分类问题
        # print((info.split(' ')[-1]))
        # if (int(info.split(' ')[-1])==0):
        #     train_image_label.append(0)
        #
        # else:
        #     train_image_label.append(1)
        # train_image_label.append(int(info.split(' ')[-1]))
    print(train_image_dir)
    print(train_image_label)

    return train_image_dir, train_image_label



def read_in_all_images(address_list, shuffle=True, is_random_label = False):
    """
    This function reads all training or validation data, shuffles them if needed, and returns the
    images and the corresponding labels as numpy arrays

    :param address_list: a list of paths of cPickle files
    :return: concatenated numpy array of data and labels. Data are in 4D arrays: [num_images,
    image_height, image_width, image_depth] and labels are in 1D arrays: [num_images]
    """
    data = np.array([]).reshape([0, IMG_WIDTH * IMG_HEIGHT * IMG_DEPTH])
    label = np.array([])

    for address in address_list:
        print ('Reading images from ' + address)
        batch_data, batch_label = _read_one_batch(address, is_random_label)
        # Concatenate along axis 0 by default
        data = np.concatenate((data, batch_data))
        label = np.concatenate((label, batch_label))

    num_data = len(label) #数据集个数即为样本个数

    # This reshape order is really important. Don't change
    # Reshape is correct. Double checked
    data = data.reshape((num_data, IMG_HEIGHT * IMG_WIDTH, IMG_DEPTH), order='F')
    data = data.reshape((num_data, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH))


    if shuffle is True:
        print ('Shuffling')
        order = np.random.permutation(num_data)
        data = data[order, ...]
        label = label[order]

    data = data.astype(np.float32)
    return data, label


def horizontal_flip(image, axis):
    '''
    注意输入是一张图片三维矩阵
    Flip an image at 50% possibility
    :param image: a 3 dimensional numpy array representing an image
    :param axis: 0 for vertical flip and 1 for horizontal flip
    :return: 3D image after flip
    '''
    flip_prop = np.random.randint(low=0, high=2)
    if flip_prop == 0:
        image = cv2.flip(image, axis)

    return image


def whitening_image(image_np):
    '''
    Performs per_image_whitening
    :param image_np: a 4D numpy array representing a batch of images
    :return: the image numpy array after whitened
    '''
    for i in range(len(image_np)):
        mean = np.mean(image_np[i, ...])
        # Use adjusted standard deviation here, in case the std == 0.
        std = np.max([np.std(image_np[i, ...]), 1.0/np.sqrt(IMG_HEIGHT * IMG_WIDTH * IMG_DEPTH)])
        image_np[i,...] = (image_np[i, ...] - mean) / std
    return image_np

def random_distort_image(image):
    rotate_image2 = ndimage.rotate(image, random.randint(1, 30))
    rotate_image2 = cv2.resize(rotate_image2, (list(image.shape)[1], list(image.shape)[0]))
    return rotate_image2

def random_image_rotate(image):
    if random.randint(0,1)==1:
        image=cv2.flip(image,0)
    if random.randint(0,1)==1:
        image=cv2.flip(image,1)
    if random.randint(0,1)==1:
        image=random_distort_image(image)
    return  image

def random_shuffle_RGB(batch_data):
    for img in batch_data:
        a=np.arange(0,3)
        np.random.shuffle(a)
        temp=img
        img[:,:,0]=temp[:,:,a[0]]
        img[:,:,1]=temp[:,:,a[1]]
        img[:,:,2]=temp[:,:,a[2]]
    # (2, 448, 336, 3) 而不是（2,3,448,336）所以不能用



def random_crop_and_flip(batch_data, padding_size):
    '''
    Helper to random crop and random flip a batch of images
    :param padding_size: int. how many layers of 0 padding was added to each side
    :param batch_data: a 4D batch array
    :return: randomly cropped and flipped image
    '''
    pad_width = ((0, 0), (padding_size, padding_size), (padding_size, padding_size), (0, 0))
    batch_data = np.pad(batch_data, pad_width=pad_width, mode='constant', constant_values=0)
    cropped_batch = np.zeros(len(batch_data) * IMG_HEIGHT * IMG_WIDTH * IMG_DEPTH).reshape(
        len(batch_data), IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH)

    for i in range(len(batch_data)):
        x_offset = np.random.randint(low=0, high=2 * padding_size, size=1)[0]
        y_offset = np.random.randint(low=0, high=2 * padding_size, size=1)[0]
        cropped_batch[i, ...] = batch_data[i, ...][x_offset:x_offset+IMG_HEIGHT,
                      y_offset:y_offset+IMG_WIDTH, :]

        cropped_batch[i, ...] = horizontal_flip(image=cropped_batch[i, ...], axis=1)

    return cropped_batch


def prepare_train_data(padding_size):
    '''
    Read all the train data into numpy array and add padding_size of 0 paddings on each side of the
    image
    :param padding_size: int. how many layers of zero pads to add on each side?
    :return: all the train data and corresponding labels ;2 1-D array
    '''
    #path_list = []
    #for i in range(1, NUM_TRAIN_BATCH+1):
    #  path_list.append(full_data_dir +'train_'+ str(i))
    #data, label = read_in_all_images(path_list, is_random_label=TRAIN_RANDOM_LABEL)
    data, label = My_read_in_all_images(is_train_image=True)
    #获得的数据实际是图片的地址（相对）和label

#以下代码添加到读取图片之后
   # pad_width = ((0, 0), (padding_size, padding_size), (padding_size, padding_size), (0, 0))
    #data = np.pad(data, pad_width=pad_width, mode='constant', constant_values=0)
    
    return data, label


def read_validation_data():
    '''
    Read in validation data. Whitening at the same time
    :return: Validation image data as 1D numpy array. Validation labels as 1D numpy array
    2 1-D array
    '''
    # path_list = []
    # for i in range(1, NUM_TRAIN_BATCH+1):
    #   path_list.append(full_data_dir +'valid_'+ str(i))
    #  validation_array, validation_labels =
# read_in_all_images([vali_dir],
    #                                                  is_random_label=
    validation_array, validation_labels = My_read_in_all_images(is_train_image=False)

    #图片读取之后再处理
    #validation_array = whitening_image(validation_array)

    return validation_array, validation_labels


