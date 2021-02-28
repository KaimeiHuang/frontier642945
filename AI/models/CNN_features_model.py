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
from tensorflow import keras

from tensorflow.keras import losses
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
import pandas as pd

from tensorflow.keras.layers import Conv2D,Dense,Dropout,Activation,BatchNormalization,MaxPooling2D,Flatten
from keras.models import Sequential
from keras import regularizers
from keras.regularizers import l1,l2
###
import tensorflow.keras.backend as K
# tf.disable_v2_behavior()
#from keras import backend as K
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model
#from keras.applications import ResNet50
#from .models import VGG19
from .VGG19 import VGG19
from .ResNet import ResNet

def CNN_features_model(image_sizes_list, label_types, activation_function, init_mode, **kwargs):
    ###features_size
    classes = label_types
    image_sizes_list_list = image_sizes_list.split(',')
    width = int(image_sizes_list_list[0])
    height = int(image_sizes_list_list[1])
    channel = int(image_sizes_list_list[2])
    inputShape = (width, height, channel)

    ###
    parameters_list = kwargs["parameters_list"]
    print("parameters_list:",parameters_list)
    parameters_dict = eval(parameters_list)
    #reg = parameters_dict["reg"]
    #print("reg:", reg)
    optimizer = parameters_dict["optimizer"]
    loss = parameters_dict["loss"]
    metrics_type = parameters_dict["metrics_type"]
    learning_rate = float(parameters_dict["learning_rate"])
    ###

    other_feature_name_strings=kwargs["other_feature_name_strings"] 
    feature_num = len(other_feature_name_strings.split(',')) - 1
    if True:
        input = keras.layers.Input(shape=inputShape)
        K.set_learning_phase(0)
        Inp = Input(inputShape)
        #base_model = ResNet50(weights='imagenet', include_top=False,input_shape=inputShape, )
        #base_model = ResNet50( include_top=False, input_shape=inputShape, )
        print("parameters_list:", parameters_list)
        print("image_sizes_list:",image_sizes_list)

        #base_model = VGG19(image_sizes_list=image_sizes_list,label_types=classes,parameters_list=parameters_list,activation_function='relu', init_mode=init_mode)
        #CNN_output_2=base_model.get_layer(name="fc2").output

        base_model = ResNet(image_sizes_list=image_sizes_list,label_types=classes,parameters_list=parameters_list,activation_function='relu', init_mode=init_mode)
        CNN_output_2=base_model.get_layer(name="avg_pool").output
        
        #用于将输入转换为密集层处理的正确形状。实际上，base_model.output tensor 的形态有 dim_ordering="th"（对应样本...通道）或者 dim_ordering="tf"（对应样本，通道，行，列），但是密集层需要将其调整为（样本，通道）GlobalAveragePooling2D 按（行，列）平均。所以如果你看最后四层（include_top=True），你会看到这些形状：
        K.set_learning_phase(1)
        print("Inp: ", Inp)
        ######
        CNN_output = base_model.output
        ######


    other_feature_input = Input(shape=(feature_num,), name='aux_input')
    
    x = keras.layers.concatenate([CNN_output_2, other_feature_input])
    

    # 堆叠多个全连接网络层
    x = Dense(64, activation='relu')(x)
    #x = Dense(64, activation='relu')(x)
    #x = Dense(64, activation='relu')(x)

    # 最后添加主要的逻辑回归层
    #main_output = Dense(units=2, activation='sigmoid', name='main_output')(x)     
    main_output = Dense(units=2, activation='softmax', name='main_output')(x)     
    #model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])
    model = Model(inputs=[*base_model.inputs, other_feature_input], outputs=[main_output, CNN_output])
    #predictions = SVM(CNN_output,features)
    #predictions = Dense(num_classes, activation='softmax')(x)
    #model = Model(inputs=Inp, outputs=predictions) 
    ###
    ##rmsprop loss=binary_crossentropy
    model.compile(optimizer=optimizer,
              loss={'main_output':loss, 'predictions': loss},
              loss_weights={ 'main_output': 0.2, 'predictions':0.2})
    #########
    if False:
        model.compile(optimizer='rmsprop', loss='binary_crossentropy',
              loss_weights=[1., 0.2])

        model.fit([headline_data, additional_data], [labels, labels],
          epochs=50, batch_size=32)
    #########

    #########
    elif False:
        #由于输入和输出均被命名了（在定义时传递了一个 name 参数），我们也可以通过以下方式编译模型：

        model.compile(optimizer='rmsprop',
              loss={'main_output': 'binary_crossentropy', 'aux_output': 'binary_crossentropy'},
              loss_weights={'main_output': 1., 'aux_output': 0.2})

        # 然后使用以下方式训练：
        model.fit({'main_input': headline_data, 'aux_input': additional_data},
          {'main_output': labels, 'aux_output': labels},
          epochs=50, batch_size=32)
    ########
     

    return model

def fit(**kwargs):
    print("in model fit of CNN_feature")
    model = kwargs["model"]
    other_feature_list_list = np.array(kwargs["other_feature_list_list"])
    parameters_dict = eval(kwargs["parameters_list"])
    epochs = parameters_dict["epochs"]
    batch_size = parameters_dict["batch_size"]

    print("other_feature_list_list:",other_feature_list_list)
    print("length of kwargs[Y_train]:",len(kwargs["Y_train"]))
    #model.fit({'main_input': kwargs["X_train"], 'aux_input': kwargs["other_feature_list_list"]},{'main_output': kwargs["Y_train"], 'aux_output': kwargs["Y_train"]},epochs=50, batch_size=32)
    #model.fit([np.array(kwargs["X_train"]),  other_feature_list_list], [kwargs["Y_train"], kwargs["Y_train"]], epochs=epochs, batch_size=batch_size)
    print("kwargs[Y_train]:",kwargs["Y_train"])
    model.fit([np.array(kwargs["X_train"]),  other_feature_list_list], [kwargs["Y_train"], kwargs["Y_train"]], epochs=epochs, batch_size=batch_size)


def predict(**kwargs):
    print("in the beginning of model: CNN_features_model")
    result_dir = kwargs["result_dir"]
    model = kwargs["model"]
    other_feature_list_list = np.array(kwargs["other_feature_list_list"])
    test_images = np.array(kwargs["image_data_list"])
    test_labels = np.array(kwargs["label_list"])
    image_name_list=kwargs["image_name_list"] 

    parameters_dict = eval(kwargs["parameters_list"])
    #epochs = parameters_dict["epochs"]
    batch_size = parameters_dict["batch_size"]


    probas_ = model.predict([test_images, other_feature_list_list], batch_size=batch_size)   
   
    print("probas_:",probas_)
    #probas_01 =  (probas_[0][:,1]) + "___" + str(probas_[1][:,1])
    
    #dataframe1 = pd.DataFrame({'image_name_list':image_name_list,'test_labels':test_labels1,"y_pred":y_pred1[:,1]})
    #dataframe1 = pd.DataFrame({'image_name_list':image_name_list,'test_labels':test_labels[:,1],"y_pred":zip(probas_[0][:,1],probas_[1][:,1])})
    dataframe1 = pd.DataFrame({'image_name_list':image_name_list,'test_labels':test_labels[:,1],"y_pred1":probas_[0][:,1],"y_pred2":probas_[1][:,1]})
    #####
    ###
    dataframe1.to_csv(result_dir + "proba_result.csv",index=False,sep=',')
    combination_df = [["patient_ID","labels","pred1","pred2"]]
    print("kwargs[ID_prefix_num]:",kwargs["ID_prefix_num"])
    dataframe1["patient_ID"] = [x[0:int(kwargs["ID_prefix_num"])] for x in dataframe1['image_name_list']]
    for image,df_for_this_image in dataframe1.groupby('patient_ID',as_index=True):
        print(df_for_this_image)
        temp_list=[image, df_for_this_image["test_labels"].min(), df_for_this_image['y_pred1'].max(), df_for_this_image['y_pred2'].mean()]
        combination_df.append(temp_list)
        #     df_for_this_image.sort_values('y_pred',inplace=True, ascending=False)
        #     print(df_for_this_image.iloc[:10])
        pd.DataFrame(combination_df).to_csv(result_dir + "proba_result_combination.csv",index=False,sep=',')

    ###

    #####
    return dataframe1

def save(**kwargs):
    model = kwargs["model"]
    print("kwargs[save_model_address]:",kwargs["save_model_address"])
    model.save(kwargs["save_model_address"])

