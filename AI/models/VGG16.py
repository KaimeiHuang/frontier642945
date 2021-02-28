from __future__ import absolute_import, division, print_function, unicode_literals
#from skimage import io, transform
import glob
import os
import numpy as np
import time
#import xlrd
#from openpyxl import load_workbook
#from openpyxl import Workbook
#from openpyxl.writer.excel import ExcelWriter
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


