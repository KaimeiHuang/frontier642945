from __future__ import absolute_import, division, print_function, unicode_literals
#from skimage import io, transform
import glob
import os
import numpy as np
import time
import xlrd
from openpyxl import load_workbook
from openpyxl import Workbook
from openpyxl.writer.excel import ExcelWriter
import cv2
# import xlwt
import os
import tensorflow as tf

from keras import losses
# from tensorflow.keras import layers
import tensorflow.keras as keras
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from scipy import interp
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate, train_test_split
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
#from sklearn.cross_validation import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score,accuracy_score
from sklearn.metrics import precision_score,f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.model_selection import cross_val_predict
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
###
import tensorflow as tf
#import tensorflow.keras as keras
import matplotlib.pyplot as plt
from keras import regularizers
###


def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = keras.layers.Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
    x = keras.layers.BatchNormalization(axis=3, name=bn_name)(x)
    return x

def Pool(input, pool_size=(3, 3), stride=(1, 1), pool_type='avg',padding='valid',name=None):
    if str.lower(pool_type) == "avg":
        x = keras.layers.AveragePooling2D(pool_size, stride,padding=padding, name=name)(input)
    elif str.lower(pool_type) == 'max':
        x = keras.layers.MaxPooling2D(pool_size, stride,padding=padding, name=name)(input)
    return x

def identity_Block(inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):
    x = Conv2d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
        x = keras.layers.add([x, shortcut])
        return x
    else:
        x = keras.layers.add([x, inpt])
        return x

def bottleneck_Block(inpt,nb_filters,strides=(1,1),with_conv_shortcut=False):
    k1,k2,k3=nb_filters
    x = Conv2d_BN(inpt, nb_filter=k1, kernel_size=1, strides=strides, padding='same')
    x = Conv2d_BN(x, nb_filter=k2, kernel_size=3, padding='same')
    x = Conv2d_BN(x, nb_filter=k3, kernel_size=1, padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt, nb_filter=k3, strides=strides, kernel_size=1)
        x = keras.layers.add([x, shortcut])
        return x
    else:
        x = keras.layers.add([x, inpt])
        return x

def Xception(image_sizes_list,classes,parameters_list,activation_function, init_mode):
    ###
    image_sizes_list = image_sizes_list.split(',')
    parameters_dict = eval(parameters_list)
    reg=parameters_dict["reg"]
    print("reg:", reg)
    print("parameters_list:",parameters_list,"type of parameters_list:",type(parameters_list))
    width = int(image_sizes_list[0])
    height = int(image_sizes_list[1])
    channel = int(image_sizes_list[2])
    print("width:",width,"height:",height,"channel:",channel,"image_sizes_list",image_sizes_list)
    ###

    print("width:",width,"height:",height,"channel:",channel,"classes:",classes,"parameters_list:",parameters_list,"activation_function:",activation_function,"init_mode:",init_mode)
    model = keras.applications.xception.Xception(include_top=True, weights=None, input_tensor=None, input_shape=(width,height,channel), pooling=None, classes=2)
    return model




def conv_block(x, nb_filters, nb_row, nb_col, strides=(1, 1), padding='same', use_bias=False):
    x = keras.layers.Conv2D(filters=nb_filters,
               kernel_size=(nb_row, nb_col),
               strides=strides,
               padding=padding,
               use_bias=use_bias)(x)
    x = keras.layers.BatchNormalization(axis=-1, momentum=0.9997, scale=False)(x)
    x = keras.layers.Activation("relu")(x)
    return x


def stem(x_input):
    x = conv_block(x_input, 32, 3, 3, strides=(2, 2), padding='valid')
    x = conv_block(x, 32, 3, 3, padding='valid')
    x = conv_block(x, 64, 3, 3)

    x1 = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)
    x2 = conv_block(x, 96, 3, 3, strides=(2, 2), padding='valid')

    x = keras.layers.concatenate([x1, x2], axis=-1)

    x1 = conv_block(x, 64, 1, 1)
    x1 = conv_block(x1, 96, 3, 3, padding='valid')

    x2 = conv_block(x, 64, 1, 1)
    x2 = conv_block(x2, 64, 7, 1)
    x2 = conv_block(x2, 64, 1, 7)
    x2 = conv_block(x2, 96, 3, 3, padding='valid')

    x = keras.layers.concatenate([x1, x2], axis=-1)

    x1 = conv_block(x, 192, 3, 3, strides=(2, 2), padding='valid')
    x2 = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)

    merged_vector = keras.layers.concatenate([x1, x2], axis=-1)
    return merged_vector


def inception_A(x_input):
    """35*35 卷积块"""
    averagepooling_conv1x1 = keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(
        x_input)  # 35 * 35 * 192
    averagepooling_conv1x1 = conv_block(averagepooling_conv1x1, 96, 1, 1)  # 35 * 35 * 96

    conv1x1 = conv_block(x_input, 96, 1, 1)  # 35 * 35 * 96

    conv1x1_3x3 = conv_block(x_input, 64, 1, 1)  # 35 * 35 * 64
    conv1x1_3x3 = conv_block(conv1x1_3x3, 96, 3, 3)  # 35 * 35 * 96

    conv3x3_3x3 = conv_block(x_input, 64, 1, 1)  # 35 * 35 * 64
    conv3x3_3x3 = conv_block(conv3x3_3x3, 96, 3, 3)  # 35 * 35 * 96
    conv3x3_3x3 = conv_block(conv3x3_3x3, 96, 3, 3)  # 35 * 35 * 96

    merged_vector = keras.layers.concatenate([averagepooling_conv1x1, conv1x1, conv1x1_3x3, conv3x3_3x3],
                                axis=-1)  # 35 * 35 * 384
    return merged_vector


def inception_B(x_input):
    """17*17 卷积块"""

    averagepooling_conv1x1 = keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x_input)
    averagepooling_conv1x1 = conv_block(averagepooling_conv1x1, 128, 1, 1)

    conv1x1 = conv_block(x_input, 384, 1, 1)

    conv1x7_1x7 = conv_block(x_input, 192, 1, 1)
    conv1x7_1x7 = conv_block(conv1x7_1x7, 224, 1, 7)
    conv1x7_1x7 = conv_block(conv1x7_1x7, 256, 1, 7)

    conv2_1x7_7x1 = conv_block(x_input, 192, 1, 1)
    conv2_1x7_7x1 = conv_block(conv2_1x7_7x1, 192, 1, 7)
    conv2_1x7_7x1 = conv_block(conv2_1x7_7x1, 224, 7, 1)
    conv2_1x7_7x1 = conv_block(conv2_1x7_7x1, 224, 1, 7)
    conv2_1x7_7x1 = conv_block(conv2_1x7_7x1, 256, 7, 1)

    merged_vector = keras.layers.concatenate([averagepooling_conv1x1, conv1x1, conv1x7_1x7, conv2_1x7_7x1], axis=-1)
    return merged_vector


def inception_C(x_input):
    """8*8 卷积块"""
    averagepooling_conv1x1 = keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(x_input)
    averagepooling_conv1x1 = conv_block(averagepooling_conv1x1, 256, 1, 1)

    conv1x1 = conv_block(x_input, 256, 1, 1)

    # 用 1x3 和 3x1 替代 3x3
    conv3x3_1x1 = conv_block(x_input, 384, 1, 1)
    conv3x3_1 = conv_block(conv3x3_1x1, 256, 1, 3)
    conv3x3_2 = conv_block(conv3x3_1x1, 256, 3, 1)

    conv2_3x3_1x1 = conv_block(x_input, 384, 1, 1)
    conv2_3x3_1x1 = conv_block(conv2_3x3_1x1, 448, 1, 3)
    conv2_3x3_1x1 = conv_block(conv2_3x3_1x1, 512, 3, 1)
    conv2_3x3_1x1_1 = conv_block(conv2_3x3_1x1, 256, 3, 1)
    conv2_3x3_1x1_2 = conv_block(conv2_3x3_1x1, 256, 1, 3)

    merged_vector = keras.layers.concatenate(
        [averagepooling_conv1x1, conv1x1, conv3x3_1, conv3x3_2, conv2_3x3_1x1_1, conv2_3x3_1x1_2], axis=-1)
    return merged_vector


def reduction_A(x_input, k=192, l=224, m=256, n=384):
    """Architecture of a 35 * 35 to 17 * 17 Reduction_A block."""
    maxpool = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x_input)

    conv3x3 = conv_block(x_input, n, 3, 3, strides=(2, 2), padding='valid')

    conv2_3x3 = conv_block(x_input, k, 1, 1)
    conv2_3x3 = conv_block(conv2_3x3, l, 3, 3)
    conv2_3x3 = conv_block(conv2_3x3, m, 3, 3, strides=(2, 2), padding='valid')

    merged_vector = keras.layers.concatenate([maxpool, conv3x3, conv2_3x3], axis=-1)
    return merged_vector


def reduction_B(x_input):
    """Architecture of a 17 * 17 to 8 * 8 Reduction_B block."""

    maxpool = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x_input)

    conv3x3 = conv_block(x_input, 192, 1, 1)
    conv3x3 = conv_block(conv3x3, 192, 3, 3, strides=(2, 2), padding='valid')

    conv1x7_7x1_3x3 = conv_block(x_input, 256, 1, 1)
    conv1x7_7x1_3x3 = conv_block(conv1x7_7x1_3x3, 256, 1, 7)
    conv1x7_7x1_3x3 = conv_block(conv1x7_7x1_3x3, 320, 7, 1)
    conv1x7_7x1_3x3 = conv_block(conv1x7_7x1_3x3, 320, 3, 3, strides=(2, 2), padding='valid')

    merged_vector = keras.layers.concatenate([maxpool, conv3x3, conv1x7_7x1_3x3], axis=-1)
    return merged_vector

def Inception_V4(width,height,channel,classes,times,L1,L2,F1,F2,F3):
    x_input = keras.layers.Input(shape=(width,height,channel))
    # Stem
    x = stem(x_input)  # 35 x 35 x 384
    # 4 x Inception_A
    for i in range(4):
        x = inception_A(x)  # 35 x 35 x 384
    # Reduction_A
    x = reduction_A(x, k=192, l=224, m=256, n=384)  # 17 x 17 x 1024
    # 7 x Inception_B
    for i in range(7):
        x = inception_B(x)  # 17 x 17 x1024
    # Reduction_B
    x = reduction_B(x)  # 8 x 8 x 1536
    # Average Pooling
    x = keras.layers.AveragePooling2D(pool_size=(8, 8))(x)  # 1536
    # dropout
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Flatten()(x)  # 1536
    # 全连接层
    x = keras.layers.Dense(units=classes, activation='softmax')(x)
    model = keras.Model(inputs=x_input, outputs=x, name='Inception-V4')
    return model

def resnet_v1_stem(x_input):
    x = keras.layers. Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='valid')(x_input)
    x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='valid')(x)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='valid')(x)
    x = keras.layers.Conv2D(80, (1, 1), activation='relu', padding='same')(x)
    x = keras.layers.Conv2D(192, (3, 3), activation='relu', padding='valid')(x)
    x = keras.layers.Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='valid')(x)
    x = keras.layers.BatchNormalization(axis=-1)(x)
    x = keras.layers.Activation('relu')(x)
    return x

def inception_resnet_v1_C(x_input, scale_residual=True):
    cr1 = keras.layers.Conv2D(192, (1, 1), activation='relu', padding='same')(x_input)

    cr2 = keras.layers.Conv2D(192, (1, 1), activation='relu', padding='same')(x_input)
    cr2 = keras.layers.Conv2D(192, (1, 3), activation='relu', padding='same')(cr2)
    cr2 = keras.layers.Conv2D(192, (3, 1), activation='relu', padding='same')(cr2)

    merged_vector = keras.layers.concatenate([cr1, cr2], axis=-1)

    cr = keras.layers.Conv2D(1792, (1, 1), activation='relu', padding='same')(merged_vector)

    if scale_residual:
        cr = keras.layers.Lambda(lambda x: 0.1 * x)(cr)
    x = keras.layers.add([x_input, cr])
    x = keras.layers.BatchNormalization(axis=-1)(x)
    x = keras.layers.Activation('relu')(x)
    return x


def reduction_resnet_B(x_input):

    rb1 = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x_input)

    rb2 = keras.layers.Conv2D(256, (1, 1), activation='relu', padding='same')(x_input)
    rb2 = keras.layers.Conv2D(384, (3, 3), strides=(2, 2), activation='relu', padding='valid')(rb2)

    rb3 = keras.layers.Conv2D(256, (1, 1), activation='relu', padding='same')(x_input)
    rb3 =keras.layers. Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='valid')(rb3)

    rb4 =keras.layers. Conv2D(256, (1, 1), activation='relu', padding='same')(x_input)
    rb4 = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(rb4)
    rb4 = keras.layers.Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='valid')(rb4)

    merged_vector = keras.layers.concatenate([rb1, rb2, rb3, rb4], axis=-1)

    x =keras.layers. BatchNormalization(axis=-1)(merged_vector)
    x = keras.layers.Activation('relu')(x)
    return x




def resnet_v2_stem(x_input):
    '''The stem of the pure Inception-v4 and Inception-ResNet-v2 networks. This is input part of those networks.'''

    # Input shape is 299 * 299 * 3 (Tensorflow dimension ordering)
    x = keras.layers.Conv2D(32, (3, 3), activation="relu", strides=(2, 2))(x_input)  # 149 * 149 * 32
    x = keras.layers.Conv2D(32, (3, 3), activation="relu")(x)  # 147 * 147 * 32
    x = keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)  # 147 * 147 * 64

    x1 = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    x2 = keras.layers.Conv2D(96, (3, 3), activation="relu", strides=(2, 2))(x)

    x = keras.layers.concatenate([x1, x2], axis=-1)  # 73 * 73 * 160

    x1 = keras.layers.Conv2D(64, (1, 1), activation="relu", padding="same")(x)
    x1 = keras.layers.Conv2D(96, (3, 3), activation="relu")(x1)

    x2 = keras.layers.Conv2D(64, (1, 1), activation="relu", padding="same")(x)
    x2 = keras.layers.Conv2D(64, (7, 1), activation="relu", padding="same")(x2)
    x2 = keras.layers.Conv2D(64, (1, 7), activation="relu", padding="same")(x2)
    x2 = keras.layers.Conv2D(96, (3, 3), activation="relu", padding="valid")(x2)

    x = keras.layers.concatenate([x1, x2], axis=-1)  # 71 * 71 * 192

    x1 = keras.layers.Conv2D(192, (3, 3), activation="relu", strides=(2, 2))(x)

    x2 = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = keras.layers.concatenate([x1, x2], axis=-1)  # 35 * 35 * 384

    x = keras.layers.BatchNormalization(axis=-1)(x)
    x = keras.layers.Activation("relu")(x)
    return x


def inception_resnet_A(x_input,n=256, scale_residual=True):
    '''Architecture of Inception_ResNet_A block which is a 35 * 35 grid module.'''
    ar1 = keras.layers.Conv2D(32, (1, 1), activation="relu", padding="same")(x_input)

    ar2 = keras.layers.Conv2D(32, (1, 1), activation="relu", padding="same")(x_input)
    ar2 = keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(ar2)

    ar3 = keras.layers.Conv2D(32, (1, 1), activation="relu", padding="same")(x_input)
    ar3 = keras.layers.Conv2D(48, (3, 3), activation="relu", padding="same")(ar3)
    ar3 = keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(ar3)

    merged = keras.layers.concatenate([ar1, ar2, ar3], axis=-1)

    ar = keras.layers.Conv2D(n, (1, 1), activation="linear", padding="same")(merged)
    if scale_residual: ar = keras.layers.Lambda(lambda a: a * 0.1)(ar) # 是否缩小

    x = keras.layers.add([x_input, ar])
    x = keras.layers.BatchNormalization(axis=-1)(x)
    x = keras.layers.Activation("relu")(x)
    return x


def inception_resnet_B(x_input,n1,n2,n3,n4,n5, scale_residual=True):
    '''Architecture of Inception_ResNet_B block which is a 17 * 17 grid module.'''
    br1 = keras.layers.Conv2D(n1, (1, 1), activation="relu", padding="same")(x_input)

    br2 = keras.layers.Conv2D(n2, (1, 1), activation="relu", padding="same")(x_input)
    br2 = keras.layers.Conv2D(n3, (1, 7), activation="relu", padding="same")(br2)
    br2 = keras.layers.Conv2D(n4, (7, 1), activation="relu", padding="same")(br2)

    merged = keras.layers.concatenate([br1, br2], axis=-1)

    br = keras.layers.Conv2D(n5, (1, 1), activation="linear", padding="same")(merged)
    if scale_residual: br = keras.layers.Lambda(lambda b: b * 0.1)(br)

    x = keras.layers.add([x_input, br])
    x = keras.layers.BatchNormalization(axis=-1)(x)
    x = keras.layers.Activation("relu")(x)
    return x


def inception_resnet_v2_C(x_input, scale_residual=True):
    '''Architecture of Inception_ResNet_C block which is a 8 * 8 grid module.'''
    cr1 = keras.layers.Conv2D(192, (1, 1), activation="relu", padding="same")(x_input)

    cr2 = keras.layers.Conv2D(192, (1, 1), activation="relu", padding="same")(x_input)
    cr2 = keras.layers.Conv2D(224, (1, 3), activation="relu", padding="same")(cr2)
    cr2 = keras.layers.Conv2D(256, (3, 1), activation="relu", padding="same")(cr2)

    merged = keras.layers.concatenate([cr1, cr2], axis=-1)

    cr = keras.layers.Conv2D(2144, (1, 1), activation="linear", padding="same")(merged)
    if scale_residual: cr = keras.layers.Lambda(lambda c: c * 0.1)(cr)

    x = keras.layers.add([x_input, cr])
    x = keras.layers.BatchNormalization(axis=-1)(x)
    x = keras.layers.Activation("relu")(x)
    return x


def reduction_resnet_A(x_input, k=192, l=224, m=256, n=384):

    ra1 = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x_input)

    ra2 = keras.layers.Conv2D(n, (3, 3), activation='relu', strides=(2, 2), padding='valid')(x_input)

    ra3 = keras.layers.Conv2D(k, (1, 1), activation='relu', padding='same')(x_input)
    ra3 = keras.layers.Conv2D(l, (3, 3), activation='relu', padding='same')(ra3)
    ra3 = keras.layers.Conv2D(m, (3, 3), activation='relu', strides=(2, 2), padding='valid')(ra3)

    merged_vector = keras.layers.concatenate([ra1, ra2, ra3], axis=-1)

    x = keras.layers.BatchNormalization(axis=-1)(merged_vector)
    x = keras.layers.Activation('relu')(x)
    return x


def reduction_resnet_v2_B(x_input):
    '''Architecture of a 17 * 17 to 8 * 8 Reduction_ResNet_B block.'''
    rbr1 =keras.layers. MaxPooling2D((3, 3), strides=(2, 2), padding="valid")(x_input)

    rbr2 = keras.layers.Conv2D(256, (1, 1), activation="relu", padding="same")(x_input)
    rbr2 = keras.layers.Conv2D(384, (3, 3), activation="relu", strides=(2, 2))(rbr2)

    rbr3 = keras.layers.Conv2D(256, (1, 1), activation="relu", padding="same")(x_input)
    rbr3 = keras.layers.Conv2D(288, (3, 3), activation="relu", strides=(2, 2))(rbr3)

    rbr4 = keras.layers.Conv2D(256, (1, 1), activation="relu", padding="same")(x_input)
    rbr4 = keras.layers.Conv2D(288, (3, 3), activation="relu", padding="same")(rbr4)
    rbr4 = keras.layers.Conv2D(320, (3, 3), activation="relu", strides=(2, 2))(rbr4)

    merged = keras.layers.concatenate([rbr1, rbr2, rbr3, rbr4], axis=-1)
    rbr = keras.layers.BatchNormalization(axis=-1)(merged)
    rbr = keras.layers.Activation("relu")(rbr)
    return rbr

def Inception_resnet_v1(width,height,channel,classes,times,L1,L2,F1,F2,F3):
    x_input = keras.layers.Input(shape=(width,height,channel))
    x = resnet_v1_stem(x_input)

    # 5 x inception_resnet_v1_A
    for i in range(5):
        x = inception_resnet_A(x,256, scale_residual=False)

    # reduction_resnet_A
    x = reduction_resnet_A(x, k=192, l=192, m=256, n=384)

    # 10 x inception_resnet_v1_B
    for i in range(10):
        x = inception_resnet_B(x,128,128,128,128,896, scale_residual=True)

    # Reduction B
    x = reduction_resnet_B(x)

    # 5 x Inception C
    for i in range(5):
        x = inception_resnet_v1_C(x, scale_residual=True)

    # Average Pooling
    x = keras.layers.AveragePooling2D(pool_size=(8, 8))(x)

    # dropout
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(units=classes, activation='softmax')(x)

    model = keras.Model(inputs=x_input, outputs=x, name='Inception-Resnet-v1')
    return model

def Inception_resnet_v2(width,height,channel,classes,times,L1,L2,F1,F2,F3,scale=True):

    x_input = keras.layers.Input(shape=(width,height,channel))
    x = resnet_v2_stem(x_input)  # Output: 35 * 35 * 256

    # 5 x Inception A
    for i in range(5):
        x = inception_resnet_A(x, 384,scale_residual=scale)
        # Output: 35 * 35 * 256

    # Reduction A
    x = reduction_resnet_A(x, k=256, l=256, m=384, n=384)  # Output: 17 * 17 * 896

    # 10 x Inception B
    for i in range(10):
        x = inception_resnet_B(x, 192,128,160,192,1152,scale_residual=scale)
        # Output: 17 * 17 * 896

    # Reduction B
    x = reduction_resnet_v2_B(x)  # Output: 8 * 8 * 1792

    # 5 x Inception C
    for i in range(5):
        x = inception_resnet_v2_C(x, scale_residual=scale)
        # Output: 8 * 8 * 1792

    # Average Pooling
    x = keras.layers.AveragePooling2D((8, 8))(x)  # Output: 1792

    # Dropout
    x = keras.layers.Dropout(0.2)(x)  # Keep dropout 0.2 as mentioned in the paper
    x = keras.layers.Flatten()(x)  # Output: 1792

    # Output layer
    output = keras.layers.Dense(units=classes, activation="softmax")(x)  # Output: 10000

    model = keras.Model(x_input, output, name="Inception-ResNet-v2")
    return model

def _group_conv(x, filters, kernel, stride, groups):

    channel_axis = 1 if keras.backend.image_data_format() == 'channels_first' else -1
    in_channels = keras.backend.int_shape(x)[channel_axis]

    # number of input channels per group
    nb_ig = in_channels // groups
    # number of output channels per group
    nb_og = filters // groups

    gc_list = []
    # Determine whether the number of filters is divisible by the number of groups
    assert filters % groups == 0

    for i in range(groups):
        if channel_axis == -1:
            x_group = keras.layers.Lambda(lambda z: z[:, :, :, i * nb_ig: (i + 1) * nb_ig])(x)
        else:
            x_group = keras.layers.Lambda(lambda z: z[:, i * nb_ig: (i + 1) * nb_ig, :, :])(x)
        gc_list.append(keras.layers.Conv2D(filters=nb_og, kernel_size=kernel, strides=stride,
                              padding='same', use_bias=False)(x_group))

    return keras.layers.Concatenate(axis=channel_axis)(gc_list)


def _channel_shuffle(x, groups):

    if keras.backend.image_data_format() == 'channels_last':
        height, width, in_channels = keras.backend.int_shape(x)[1:]
        channels_per_group = in_channels // groups
        pre_shape = [-1, height, width, groups, channels_per_group]
        dim = (0, 1, 2, 4, 3)
        later_shape = [-1, height, width, in_channels]
    else:
        in_channels, height, width = keras.backend.int_shape(x)[1:]
        channels_per_group = in_channels // groups
        pre_shape = [-1, groups, channels_per_group, height, width]
        dim = (0, 2, 1, 3, 4)
        later_shape = [-1, in_channels, height, width]

    x = keras.layers.Lambda(lambda z: keras.backend.reshape(z, pre_shape))(x)
    x = keras.layers.Lambda(lambda z: keras.backend.permute_dimensions(z, dim))(x)
    x = keras.layers.Lambda(lambda z: keras.backend.reshape(z, later_shape))(x)

    return x


def _shufflenet_unit(inputs, filters, kernel, stride, groups, stage, bottleneck_ratio=0.25):
    channel_axis = 1 if keras.backend.image_data_format() == 'channels_first' else -1
    in_channels = keras.backend.int_shape(inputs)[channel_axis]
    bottleneck_channels = int(filters * bottleneck_ratio)

    if stage == 2:
        x = keras.layers.Conv2D(filters=bottleneck_channels, kernel_size=kernel, strides=1,
                   padding='same', use_bias=False)(inputs)
    else:
        x = _group_conv(inputs, bottleneck_channels, (1, 1), 1, groups)
    x = keras.layers.BatchNormalization(axis=channel_axis)(x)
    x = keras.layers.ReLU()(x)

    x = _channel_shuffle(x, groups)

    x = keras.layers.DepthwiseConv2D(kernel_size=kernel, strides=stride, depth_multiplier=1,
                        padding='same', use_bias=False)(x)
    x = keras.layers.BatchNormalization(axis=channel_axis)(x)

    if stride == 2:
        x = _group_conv(x, filters - in_channels, (1, 1), 1, groups)
        x = keras.layers.BatchNormalization(axis=channel_axis)(x)
        avg = keras.layers.AveragePooling2D(pool_size=(3, 3), strides=2, padding='same')(inputs)
        x = keras.layers.Concatenate(axis=channel_axis)([x, avg])
    else:
        x = _group_conv(x, filters, (1, 1), 1, groups)
        x = keras.layers.BatchNormalization(axis=channel_axis)(x)
        x = keras.layers.add([x, inputs])

    return x


def _stage(x, filters, kernel, groups, repeat, stage):
    x = _shufflenet_unit(x, filters, kernel, 2, groups, stage)

    for i in range(1, repeat):
        x = _shufflenet_unit(x, filters, kernel, 1, groups, stage)

    return x


def ShuffleNet(width,height,channel,classes,times,L1,L2,F1,F2,F3):
    inputs = keras.layers.Input(shape=(width,height,channel))

    x = keras.layers.Conv2D(24, (3, 3), strides=2, padding='same', use_bias=True, activation='relu')(inputs)
    x = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

    x = _stage(x, filters=384, kernel=(3, 3), groups=8, repeat=4, stage=2)
    x = _stage(x, filters=768, kernel=(3, 3), groups=8, repeat=8, stage=3)
    x = _stage(x, filters=1536, kernel=(3, 3), groups=8, repeat=4, stage=4)

    x = keras.layers.GlobalAveragePooling2D()(x)

    x = keras.layers.Dense(classes)(x)
    predicts = keras.layers.Activation('softmax')(x)

    model = keras.Model(inputs, predicts)

    return model
