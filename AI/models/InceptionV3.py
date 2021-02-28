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
#import tensorflow.keras as keras
import matplotlib.pyplot as plt
from keras import regularizers
#from keras.layers import *

from tensorflow.keras.layers import Conv2D,Dense,Dropout,Activation,BatchNormalization,MaxPooling2D,Flatten
from keras.models import Sequential
from keras import regularizers
from keras.regularizers import l1,l2
###
import tensorflow.keras.backend as K
# tf.disable_v2_behavior()
#from keras import backend as K

def LeNet(image_sizes_list,classes,parameters_list,activation_function, init_mode):
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

    #model = Sequential()
    model = keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32,(5,5),strides=(1,1),input_shape=(width,height,channel),padding='valid',activation='relu',kernel_initializer=init_mode))  ### 'uniform'
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(keras.layers.Conv2D(64,(5,5),strides=(1,1),padding='valid',activation='relu',kernel_initializer='uniform'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(100,activation='relu'))
    model.add(keras.layers.Dense(classes,activation='softmax'))
    return model

def CNN3(image_sizes_list,classes,parameters_list,activation_function, init_mode):

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

    model = keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(8, (5, 5), activation='relu', input_shape=(width,height,channel)))
    for i in range(3):
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None)) ##benben
        model.add(keras.layers.Conv2D(8, (3, 3), activation='relu'))
    # model.add(keras.layers.MaxPooling2D((2, 2)))
    # model.add(keras.layers.Conv2D(8, (3, 3), activation='relu'))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units=classes, activation='softmax',kernel_regularizer=regularizers.l2(0.001),activity_regularizer=regularizers.l1(0.01)))
    print(model.summary())
    return model

#def AlexNet(width,height,depth,classes,reg=0.0002):
#def AlexNet(image_sizes_list,classes,parameters_list,activation_function, init_mode):
def AlexNet(**kwargs):
    image_sizes_list = kwargs["image_sizes_list"]
    classes = kwargs["label_types"]
    parameters_list = kwargs["parameters_list"]
    activation_function = kwargs["activation_function"]
    init_mode = kwargs["init_mode"]
    ###
    image_sizes_list = image_sizes_list.split(',')
    print("parameters_list: ",parameters_list)
    parameters_dict = eval(parameters_list) 
    reg=parameters_dict["reg"]        
    print("reg:", reg)
    print("parameters_list:",parameters_list,"type of parameters_list:",type(parameters_list))
    width = int(image_sizes_list[0])
    height = int(image_sizes_list[1])
    channel = int(image_sizes_list[2])
    print("width:",width,"height:",height,"channel:",channel,"image_sizes_list",image_sizes_list)
    ###

    inputShape = (width,height,channel)
    model = keras.models.Sequential()
    chanDim = -1

    #if K.image_data_format() == "channels_first":
    #        inputShape = (depth,height,width)
    #        chanDim = 1
    model.add(tf.keras.layers.Conv2D(96,(11,11),strides=(4,4),input_shape=inputShape,padding="same",kernel_regularizer=l2(reg)))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(256,(5,5),padding="same",kernel_regularizer=l2(reg)))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(384,(3,3),padding="same",kernel_regularizer=l2(reg)))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))

    model.add(Conv2D(384,(3,3),padding="same",kernel_regularizer=l2(reg)))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))

    model.add(Conv2D(256,(3,3),padding="same",kernel_regularizer=l2(reg)))
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(4096,kernel_regularizer=l2(reg)))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Dense(4096,kernel_regularizer=l2(reg)))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    model.add(Dense(int(classes),kernel_regularizer=l2(reg)))
    model.add(Activation("softmax"))
    return model


def AlexNet1(image_sizes_list,classes,parameters_list,activation_function, init_mode):

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

    inputShape = (width,height,channel)
    model = keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(96,(11,11),strides=(4,4),input_shape=inputShape,padding='valid',activation='relu',kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    model.add(Conv2D(256,(5,5),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))
    model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
    model.add(Flatten())
    model.add(Dense(4096,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes,activation='softmax'))
    return model

def fit(**kwargs):
    print("in model VGG")
    model = kwargs["model"]
    print("parameters_list:",kwargs["parameters_list"])
    parameters_dict = eval(kwargs["parameters_list"])
    epochs=parameters_dict["epochs"]
    batch_size= parameters_dict["batch_size"]
 
    model.fit(kwargs["X_train"], kwargs["Y_train"], batch_size=batch_size, validation_data=(kwargs["X_train"], kwargs["Y_train"]), epochs=epochs)


def predict(**kwargs):
    model = kwargs["model"]
    probas_ = model.predict(kwargs["X_test"], batch_size=kwargs["batch_size"])
    print("probas_:",probas_)
    return probas_

def save(**kwargs):
    model = kwargs["model"]
    model.save(kwargs["save_model_address"])


#def VGG16(image_sizes_list,classes,parameters_list, activation, init_mode):
def VGG16(**kwargs):
    '''
    VGG16 model added in 20200825
    :param image_sizes_list:
    :param classes:
    :param parameters_list:
    :param activation_function:
    :param init_mode:
    :return:
    '''
    image_sizes_list = kwargs["image_sizes_list"]
    classes = kwargs["label_types"]
    activation = kwargs["activation_function"]
    init_mode = kwargs["init_mode"]
    image_sizes_list = image_sizes_list.split(',')
    parameters_list = kwargs["parameters_list"]
    parameters_dict = eval(parameters_list)
    if "activation_function" in parameters_dict.keys():
        activation = parameters_dict["activation_function"]
    else:
       activation = "relu" 
    # reg = parameters_dict["reg"]
    # print("reg:", reg)
    print("parameters_list:", parameters_list, "type of parameters_list:", type(parameters_list))
    width = int(image_sizes_list[0])
    height = int(image_sizes_list[1])
    channel = int(image_sizes_list[2])
    inputShape = (width, height, channel)
    model = keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (3, 3), input_shape=inputShape, padding='valid', activation=activation, kernel_initializer=init_mode))
    model.add(Conv2D(64, (3, 3), padding='valid', activation=activation, kernel_initializer=init_mode))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(128, (3, 3), padding='valid', activation=activation, kernel_initializer=init_mode))
    model.add(Conv2D(128, (3, 3), padding='valid', activation=activation, kernel_initializer=init_mode))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(256, (3, 3), padding='valid', activation=activation, kernel_initializer=init_mode))
    model.add(Conv2D(256, (3, 3), padding='valid', activation=activation, kernel_initializer=init_mode))
    model.add(Conv2D(256, (3, 3), padding='valid', activation=activation, kernel_initializer=init_mode))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(512, (3, 3), padding='valid', activation=activation, kernel_initializer=init_mode))
    model.add(Conv2D(512, (3, 3), padding='valid', activation=activation, kernel_initializer=init_mode))
    model.add(Conv2D(512, (3, 3), padding='valid', activation=activation, kernel_initializer=init_mode))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(512, (3, 3), padding='same', activation=activation, kernel_initializer=init_mode))
    model.add(Conv2D(512, (3, 3), padding='same', activation=activation, kernel_initializer=init_mode))
    model.add(Conv2D(512, (3, 3), padding='same', activation=activation, kernel_initializer=init_mode))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # After the  convolutional layer, it is impossible to directly connect to the Dense fully connected layer. The data of the Convolution layer needs to be flattened.
    model.add(Flatten())
    model.add(keras.layers.Dense(4096, activation=activation))
    model.add(BatchNormalization())
    model.add(keras.layers.Dense(4096, activation=activation))
    model.add(BatchNormalization())
    model.add(keras.layers.Dense(1000, activation=activation))
    model.add(BatchNormalization())
    model.add(Dense(classes, activation='softmax'))
    return model


def VGG19(image_sizes_list,label_types,parameters_list,activation_function, init_mode):
    '''
    VGG19 model added in 20200825
    :param image_sizes_list:
    :param classes:
    :param parameters_list:
    :param activation_function:
    :param init_mode:
    :return:
    '''
    image_sizes_list = image_sizes_list.split(',')
    print("parameters_list:",parameters_list)
    parameters_dict = eval(parameters_list)
    reg = parameters_dict["reg"]
    print("reg:", reg)
    print("parameters_list:", parameters_list, "type of parameters_list:", type(parameters_list))
    width = int(image_sizes_list[0])
    height = int(image_sizes_list[1])
    channel = int(image_sizes_list[2])
    inputShape = (width, height, channel)
    model = keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(64, (3, 3), input_shape=inputShape, padding='valid', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(64, (3, 3), padding='valid', activation='relu', kernel_initializer='uniform'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(128, (3, 3), padding='valid', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(128, (3, 3), padding='valid', activation='relu', kernel_initializer='uniform'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(256, (3, 3), padding='valid', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(256, (3, 3), padding='valid', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(256, (3, 3), padding='valid', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(256, (3, 3), padding='valid', activation='relu', kernel_initializer='uniform'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(keras.layers.Dense(4096, activation='relu'))
    model.add(BatchNormalization())
    model.add(keras.layers.Dense(4096, activation='relu'))
    model.add(BatchNormalization())
    model.add(keras.layers.Dense(1000, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(label_types, activation='softmax'))
    return model


class BasicConv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding):
        super(BasicConv2D, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters=filters,
                                           kernel_size=kernel_size,
                                           strides=strides,
                                           padding=padding)
        self.bn = BatchNormalization()
        self.relu = keras.layers.ReLU()

    def __call__(self, inputs, **kwargs):
        output = self.conv(inputs)
        output = self.bn(output)
        output = self.relu(output)

        return output
    

class InceptionModule_1(tf.keras.layers.Layer):
    def __init__(self, filter_num):
        super(InceptionModule_1, self).__init__()
        # branch 0
        self.conv_b0_1 = BasicConv2D(filters=64,
                                     kernel_size=(1, 1),
                                     strides=1,
                                     padding='same')

        # branch 1
        self.conv_b1_1 = BasicConv2D(filters=48,
                                     kernel_size=(1, 1),
                                     strides=1,
                                     padding='same')
        self.conv_b1_2 = BasicConv2D(filters=64,
                                     kernel_size=(5, 5),
                                     strides=1,
                                     padding='same')

        # branch 2
        self.conv_b2_1 = BasicConv2D(filters=64,
                                     kernel_size=(1, 1),
                                     strides=1,
                                     padding='same')
        self.conv_b2_2 = BasicConv2D(filters=96,
                                     kernel_size=(3, 3),
                                     strides=1,
                                     padding='same')
        self.conv_b2_3 = BasicConv2D(filters=96,
                                     kernel_size=(3, 3),
                                     strides=1,
                                     padding='same')

        # branch 3
        self.avgpool_b3_1 = keras.layers.AveragePooling2D(pool_size=(3, 3),
                                                          strides=1,
                                                          padding='same')
        self.conv_b3_2 = BasicConv2D(filters=filter_num,
                                     kernel_size=(1, 1),
                                     strides=1,
                                     padding='same')

    def __call__(self, inputs, **kwargs):
        b0 = self.conv_b0_1(inputs)

        b1 = self.conv_b1_1(inputs)
        b1 = self.conv_b1_2(b1)

        b2 = self.conv_b2_1(inputs)
        b2 = self.conv_b2_2(b2)
        b2 = self.conv_b2_3(b2)

        b3 = self.avgpool_b3_1(inputs)
        b3 = self.conv_b3_2(b3)

        output = keras.layers.concatenate([b0, b1, b2, b3], axis=3)

        return output


class InceptionModule_2(tf.keras.layers.Layer):
    def __init__(self):
        # branch 0
        self.conv_b0_1 = BasicConv2D(filters=384,
                                     kernel_size=(3, 3),
                                     strides=2,
                                     padding='same')

        # branch 1
        self.conv_b1_1 = BasicConv2D(filters=64,
                                     kernel_size=(1, 1),
                                     strides=1,
                                     padding='same')
        self.conv_b1_2 = BasicConv2D(filters=96,
                                     kernel_size=(3, 3),
                                     strides=1,
                                     padding='same')
        self.conv_b1_3 = BasicConv2D(filters=96,
                                     kernel_size=(3, 3),
                                     strides=2,
                                     padding='same')

        # branch 2
        self.maxpool_b2_1 = MaxPooling2D((3, 3),
                                         strides=2,
                                         padding='same')

    def __call__(self, inputs, **kwargs):
        b0 = self.conv_b0_1(inputs)

        b1 = self.conv_b1_1(inputs)
        b1 = self.conv_b1_2(b1)
        b1 = self.conv_b1_3(b1)

        b2 = self.maxpool_b2_1(inputs)

        output = keras.layers.concatenate([b0, b1, b2], axis=3)
        return output


class InceptionModule_3(tf.keras.layers.Layer):
    def __init__(self, filter_num):
        # branch 0
        self.conv_b0_1 = BasicConv2D(filters=192,
                                     kernel_size=(1, 1),
                                     strides=1,
                                     padding='same')

        # branch 1
        self.conv_b1_1 = BasicConv2D(filters=filter_num,
                                     kernel_size=(1, 1),
                                     strides=1,
                                     padding='same')
        self.conv_b1_2 = BasicConv2D(filters=filter_num,
                                     kernel_size=(1, 7),
                                     strides=1,
                                     padding='same')
        self.conv_b1_3 = BasicConv2D(filters=192,
                                     kernel_size=(7, 1),
                                     strides=1,
                                     padding='same')

        # branch 2
        self.conv_b2_1 = BasicConv2D(filters=filter_num,
                                     kernel_size=(1, 1),
                                     strides=1,
                                     padding='same')
        self.conv_b2_2 = BasicConv2D(filters=filter_num,
                                     kernel_size=(7, 1),
                                     strides=1,
                                     padding='same')
        self.conv_b2_3 = BasicConv2D(filters=filter_num,
                                     kernel_size=(1, 7),
                                     strides=1,
                                     padding='same')
        self.conv_b2_4 = BasicConv2D(filters=filter_num,
                                     kernel_size=(7, 1),
                                     strides=1,
                                     padding='same')
        self.conv_b2_5 = BasicConv2D(filters=192,
                                     kernel_size=(1, 7),
                                     strides=1,
                                     padding='same')

        # branch 3
        self.avgpool_b3_1 = keras.layers.AveragePooling2D(pool_size=(3, 3),
                                                          strides=1,
                                                          padding='same')
        self.conv_b3_2 = BasicConv2D(filters=192,
                                     kernel_size=(1, 1),
                                     strides=1,
                                     padding='same')

    def __call__(self, inputs, **kwargs):
        b0 = self.conv_b0_1(inputs)

        b1 = self.conv_b1_1(inputs)
        b1 = self.conv_b1_2(b1)
        b1 = self.conv_b1_3(b1)

        b2 = self.conv_b2_1(inputs)
        b2 = self.conv_b2_2(b2)
        b2 = self.conv_b2_3(b2)
        b2 = self.conv_b2_4(b2)
        b2 = self.conv_b2_5(b2)

        b3 = self.avgpool_b3_1(inputs)
        b3 = self.conv_b3_2(b3)

        output = keras.layers.concatenate([b0, b1, b2, b3], axis=3)
        return output


class InceptionModule_4(tf.keras.layers.Layer):
    def __init__(self):
        # branch 0
        self.conv_b0_1 = BasicConv2D(filters=192,
                                     kernel_size=(1, 1),
                                     strides=1,
                                     padding='same')
        self.conv_b0_2 = BasicConv2D(filters=320,
                                     kernel_size=(3, 3),
                                     strides=2,
                                     padding='same')

        # branch 1
        self.conv_b1_1 = BasicConv2D(filters=192,
                                     kernel_size=(1, 1),
                                     strides=1,
                                     padding='same')
        self.conv_b1_2 = BasicConv2D(filters=192,
                                     kernel_size=(1, 7),
                                     strides=1,
                                     padding='same')
        self.conv_b1_3 = BasicConv2D(filters=192,
                                     kernel_size=(7, 1),
                                     strides=1,
                                     padding='same')
        self.conv_b1_4 = BasicConv2D(filters=192,
                                     kernel_size=(3, 3),
                                     strides=2,
                                     padding='same')

        # branch 2
        self.maxpool_b2_1 = keras.layers.AveragePooling2D((3, 3), strides=2, padding='same')

    def __call__(self, inputs, **kwargs):
        b0 = self.conv_b0_1(inputs)
        b0 = self.conv_b0_2(b0)

        b1 = self.conv_b1_1(inputs)
        b1 = self.conv_b1_2(b1)
        b1 = self.conv_b1_3(b1)
        b1 = self.conv_b1_4(b1)

        b2 = self.maxpool_b2_1(inputs)

        output = keras.layers.concatenate([b0, b1, b2], axis=3)
        return output


class InceptionModule_5(tf.keras.layers.Layer):
    def __init__(self):
        self.conv1 = BasicConv2D(filters=320,
                                 kernel_size=(1, 1),
                                 strides=1,
                                 padding='same')
        self.conv2 = BasicConv2D(filters=384,
                                 kernel_size=(1, 1),
                                 strides=1,
                                 padding='same')
        self.conv3 = BasicConv2D(filters=384,
                                 kernel_size=(1, 3),
                                 strides=1,
                                 padding='same')
        self.conv4 = BasicConv2D(filters=384,
                                 kernel_size=(3, 1),
                                 strides=1,
                                 padding='same')
        self.conv5 = BasicConv2D(filters=448,
                                 kernel_size=(1, 1),
                                 strides=1,
                                 padding='same')
        self.conv6 = BasicConv2D(filters=384,
                                 kernel_size=(3, 3),
                                 strides=1,
                                 padding='same')
        self.conv7 = BasicConv2D(filters=192,
                                 kernel_size=(1, 1),
                                 strides=1,
                                 padding='same')
        self.avgpool = keras.layers.AveragePooling2D(pool_size=(3, 3),
                                                     strides=1,
                                                     padding='same')

    def __call__(self, inputs, **kwargs):
        b0 = self.conv1(inputs)
        
        b1 = self.conv2(inputs)
        b1_a = self.conv3(b1)
        b1_b = self.conv4(b1)
        b1 = keras.layers.concatenate([b1_a, b1_b], axis=3)
        
        b2 = self.conv5(inputs)
        b2 = self.conv6(b2)
        b2_a = self.conv3(b2)
        b2_b = self.conv4(b2)
        b2 = keras.layers.concatenate([b2_a, b2_b], axis=3)
        
        b3 = self.avgpool(inputs)
        b3 = self.conv7(b3)
        
        output = keras.layers.concatenate([b0, b1, b2, b3], axis=3)
        return output


def InceptionV3(image_sizes_list,classes,parameters_list,activation_function, init_mode):
    image_sizes_list = image_sizes_list.split(',')
    parameters_dict = eval(parameters_list)
    reg = parameters_dict["reg"]
    print("reg:", reg)
    print("parameters_list:", parameters_list, "type of parameters_list:", type(parameters_list))
    width = int(image_sizes_list[0])
    height = int(image_sizes_list[1])
    channel = int(image_sizes_list[2])
    inputShape = (width, height, channel)
    input = keras.layers.Input(shape=inputShape)

    # block_1 = keras.models.Sequential([
    #     InceptionModule_1(filter_num=32),
    #     InceptionModule_1(filter_num=64),
    #     InceptionModule_1(filter_num=64)
    # ])
    # block_2 = keras.models.Sequential([
    #     InceptionModule_2(),
    #     InceptionModule_3(filter_num=128),
    #     InceptionModule_3(filter_num=160),
    #     InceptionModule_3(filter_num=160),
    #     InceptionModule_3(filter_num=192)
    # ])
    # block_3 = keras.models.Sequential([
    #     InceptionModule_4(),
    #     InceptionModule_5(),
    #     InceptionModule_5()
    # ])

    x = BasicConv2D(filters=32,
                    kernel_size=(3, 3),
                    strides=2,
                    padding='same')(input)
    x = BasicConv2D(filters=32,
                    kernel_size=(3, 3),
                    strides=1,
                    padding='same')(x)
    x = BasicConv2D(filters=64,
                    kernel_size=(3, 3),
                    strides=1,
                    padding='same')(x)
    x = MaxPooling2D((3, 3), strides=2)(x)
    x = BasicConv2D(filters=80,
                    kernel_size=(1, 1),
                    strides=1,
                    padding='same')(x)
    x = BasicConv2D(filters=192,
                    kernel_size=(3, 3),
                    strides=1,
                    padding='same')(x)
    x = MaxPooling2D((3, 3), strides=2)(x)
    # x = block_1(x)
    # x = block_2(x)
    # x = block_3(x)
    x = InceptionModule_1(filter_num=32)(x)
    x = InceptionModule_1(filter_num=64)(x)
    x = InceptionModule_1(filter_num=64)(x)
    x = InceptionModule_2()(x)
    x = InceptionModule_3(filter_num=128)(x)
    x = InceptionModule_3(filter_num=160)(x)
    x = InceptionModule_3(filter_num=160)(x)
    x = InceptionModule_3(filter_num=192)(x)
    x = InceptionModule_4()(x)
    x = InceptionModule_5()(x)
    x = InceptionModule_5()(x)
    x = keras.layers.AveragePooling2D(pool_size=(8, 8), strides=1, padding='same')(x)
    x = Dropout(rate=0.5)(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dense(classes, activation="softmax")(x)

    model = keras.models.Model(inputs=input, outputs=x)

    return model

