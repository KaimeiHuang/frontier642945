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
import pandas as pd
import keras
from keras import losses
# from tensorflow.keras import layers
#import tensorflow.keras as keras
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
from tensorflow.keras.optimizers import SGD
from keras import optimizers

def fit(**kwargs):
    print("in model VGG")
    model = kwargs["model"]
    print("parameters_list:",kwargs["parameters_list"])
    parameters_dict = eval(kwargs["parameters_list"])
    epochs=parameters_dict["epochs"]
    batch_size= parameters_dict["batch_size"]
    learning_rate =float(parameters_dict["learning_rate"])
    print("learning_rate:",learning_rate)
    EarlyStopping_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=0.0000001, patience=5, verbose=1, restore_best_weights=True)
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir='/tmp/benlogs')
    checkpoint_filepath = kwargs["save_model_address"]
    #checkpoint_filepath = "/tmp/benlogs1"
    print("checkpoint_filepath:", checkpoint_filepath)
    #加载权重文件时候只需要写入上边被注释掉的网络结构即可
    #model.load_weights(checkpoint_filepath)
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
                                    filepath=checkpoint_filepath,
                                    save_weights_only=False,
                                    monitor='val_accuracy',
                                    mode='max',
                                    save_best_only=True)
    #model = tf.keras.models.load_model(checkpoint_filepath)
    model.fit(kwargs["X_train"], kwargs["Y_train"], batch_size=batch_size, validation_data=(kwargs["X_train"], kwargs["Y_train"]), epochs=epochs, verbose=1)
    #model.fit(kwargs["X_train"], kwargs["Y_train"], batch_size=batch_size, validation_data=(kwargs["X_train"], kwargs["Y_train"]), epochs=epochs, callbacks=[EarlyStopping_callback], verbose=1)
    model.save(kwargs["save_model_address"])
    #restored_model = tf.keras.models.load_model(kwargs["save_model_address"])

    #restored_model.fit(kwargs["X_train"], kwargs["Y_train"], batch_size=batch_size, validation_data=(kwargs["X_train"], kwargs["Y_train"]), epochs=10, callbacks=[EarlyStopping_callback], verbose=1)
    #model.load_weights(checkpoint_filepath)

def predict(**kwargs):
    result_dir = kwargs["result_dir"]
    test_images = np.array(kwargs["image_data_list"])
    test_labels = np.array(kwargs["label_list"])
    image_name_list=kwargs["image_name_list"]

    model = kwargs["model"]

    parameters_dict = eval(kwargs["parameters_list"])
    epochs=parameters_dict["epochs"]
    batch_size= parameters_dict["batch_size"]

    probas_ = model.predict(test_images, batch_size=kwargs["batch_size"])
    print("probas_:",probas_)
    print("test_labels:",test_labels)
    print("probas_[:,1]:",probas_[:,1])
    dataframe1 = pd.DataFrame({'image_name_list':image_name_list,'test_labels':test_labels[:,1],"y_pred":probas_[:,1]})

    ###
    dataframe1.to_csv(result_dir + "pred_result.csv",index=False,sep=',')
    combination_df = [["patient_ID","my_pred","labels"]]
    print("kwargs[ID_prefix_num]:",kwargs["ID_prefix_num"])
    dataframe1["patient_ID"] = [x[0:int(kwargs["ID_prefix_num"])] for x in dataframe1['image_name_list']]
    for image,df_for_this_image in dataframe1.groupby('patient_ID',as_index=True):
        print(df_for_this_image)
        temp_list=[image,df_for_this_image['y_pred'].mean(),df_for_this_image["test_labels"].min()]
        combination_df.append(temp_list)
        pd.DataFrame(combination_df).to_csv(result_dir + "pred_result_combination.csv",index=False,sep=',')

    ###
    return dataframe1

def save(**kwargs):
    model = kwargs["model"]
    model.save(kwargs["save_model_address"])


def VGG19(**kwargs):
    '''
    VGG19 model added in 20200825
    :param image_sizes_list:
    :param classes:
    :param parameters_list:
    :param activation_function:
    :param init_mode:
    :return:
    '''
    image_sizes_list = kwargs["image_sizes_list"]
    image_sizes_list = image_sizes_list.split(',')
    label_types = kwargs["label_types"]
    activation_function = kwargs["activation_function"]
    init_mode = kwargs["init_mode"]
        
 
    parameters_list = kwargs["parameters_list"]
    print("parameters_list:",parameters_list)
    parameters_dict = eval(parameters_list)
    #reg = parameters_dict["reg"]
    #print("reg:", reg)
    optimizer = parameters_dict["optimizer"]
    #optimizer = kwargs["optimizer"]
    loss = parameters_dict["loss"]
    #loss = kwargs["loss"]
    metrics_type = parameters_dict["metrics_type"]
    learning_rate = float(parameters_dict["learning_rate"])
    #metrics_type = kwargs["metrics_type"]


    print("image_sizes_list:", image_sizes_list)
    print("parameters_list:", parameters_list, "type of parameters_list:", type(parameters_list))
    width = int(image_sizes_list[0])
    height = int(image_sizes_list[1])
    channel = int(image_sizes_list[2])
    inputShape = (width, height, channel)
    print("inputShape:",inputShape)
    model = tf.keras.applications.VGG19(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=inputShape,
    pooling=None,
    classes=2,
    #classifier_activation="softmax",
    )
    #model.compile(loss=loss, optimizer=optimizer, metrics=[metrics_type])
    optimizer1 = SGD(lr=0.01, decay=1e-4, momentum=0.9, nesterov=True)  
    #optimizer1 =  tf.keras.optimizers.SGD(lr=0.01,clipnorm=1.)  
    #optimizer1 = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss=loss, optimizer=optimizer1, metrics=[metrics_type])
    #model.compile(SGD(1e-4, decay=1e-2), loss='mse')
    model.summary()
    ###

 
    return model
######

#def VGG19(image_sizes_list,label_types,parameters_list,activation_function, init_mode):
def VGG19_bak(**kwargs):
    '''
    VGG19 model added in 20200825
    :param image_sizes_list:
    :param classes:
    :param parameters_list:
    :param activation_function:
    :param init_mode:
    :return:
    '''
    image_sizes_list = kwargs["image_sizes_list"]
    label_types = kwargs["label_types"]
    parameters_list = kwargs["parameters_list"]
    activation_function = kwargs["activation_function"]
    init_mode = kwargs["init_mode"]

    image_sizes_list = image_sizes_list.split(',')
    print("parameters_list:",parameters_list)
    parameters_dict = eval(parameters_list)
    #reg = parameters_dict["reg"]
    #print("reg:", reg)
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
    model.add(keras.layers.Dense(4096, activation='relu',name="fc1"))
    model.add(BatchNormalization())
    model.add(keras.layers.Dense(4096, activation='relu',name="fc2"))
    model.add(BatchNormalization())
    #model.add(keras.layers.Dense(units=label_types, activation='softmax',name="CNN_output"))
    #model.add(keras.layers.Dense(units=2, activation='softmax',name="CNN_output"))
    model.add(keras.layers.Dense(units=2, activation='softmax',name="predictions"))
    return model


