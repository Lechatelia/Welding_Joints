import cv2
import cifar10_input
import numpy as np
from skimage import io,transform
file_list=[
    'pic1.jpg',
    'pic2.jpg'
]


def image_show(str1,image_arr):
    i=0
    for image in image_arr:
        print(image_arr.shape)
        str1=str1+str(i)
        cv2.imshow(str1,image)
        cv2.waitKey()
        i=i+1


def ds():
    train_img = np.zeros((len(file_list),448, 336, 3))
    flip1 = np.zeros((len(file_list),448, 336, 3))
    flip2 = np.zeros((len(file_list),448, 336, 3))
    rotate = np.zeros((len(file_list),448, 336, 3))
    i=0
    for file in file_list:
        img=cv2.imread(file)
        img=cv2.resize(img,(336,448))
        train_img[i]=img/256
        i = i + 1
    image_show('yuantu',train_img)
    flip1[0]=cifar10_input.random_image_rotate(train_img[0])
    flip1[1]=cifar10_input.random_image_rotate(train_img[0])
    flip2[0]=cifar10_input.random_image_rotate(train_img[0])
    flip2[1]=cifar10_input.random_image_rotate(train_img[0])
    rotate[0]=cifar10_input.random_image_rotate(train_img[0])
    rotate[1]=cifar10_input.random_image_rotate(train_img[0])

    crop=cifar10_input.random_crop_and_flip(train_img,25)
    image_show('truantu',train_img)
    image_show('flip1',flip1)
    image_show('flip2',flip2)
    image_show('rotate',rotate)
    whitening = cifar10_input.whitening_image(train_img)
    image_show('whilt',whitening)
    image_show('crop',crop)
    cifar10_input.random_shuffle_RGB(train_img)
    image_show('RGB', train_img)


def abc():
    img = cv2.imread(file_list[0])
    img = cv2.resize(img, (240, 320))
    print(type(img))
    cv2.imshow('1',img)
    cv2.waitKey()
    img=np.array(img)

def skimg():
    img=io.imread('test.jpg')
    print(str(img.shape))
    io.imshow(img)
    io.show()


if __name__ == '__main__':
    # ds()
    skimg()