#!~/.conda/envs/python36/bin/python
#coding=utf-8
import argparse
import numpy as np
#import tensorflow as tf
import os, sys
path_wd = os.path.dirname(sys.argv[0])
sys.path.append(path_wd)
from numpy import *
import scipy.misc
#import imageio
from glob import glob
#import matplotlib.image as mpimg
#from PIL import Image
from os.path import join, split
import openslide
#from openslide.deepzoom import DeepZoomGenerator
#from tensorflow_CNN import model_train
import random
import pandas as pd
import cv2
import tensorflow as  tf
#import draw_heatmap
sys.path.append("..")
import  AI.build_model as build_model
import random
from keras.utils.np_utils import to_categorical
#############
def get_label_list(image_name_list, label_address, label_list,label_name='label'):
        labels_table = pd.DataFrame(pd.read_csv(label_address,names=['file_name', 'label', 'type'], header=0))
        for image_name in image_name_list:
                #print("image_name:",image_name)
                patient_label_info = labels_table.loc[(labels_table['file_name'] == image_name), ['label']]
                if(len(patient_label_info['label'].values) == 0):
                        label_list.append(-1)
                #print("patient_label_info.shape:",patient_label_info.shape)
                #print("patient_label_info:", patient_label_info)
                label_list.append(patient_label_info['label'].values[0])
def get_patient_ID_label_list(label_address, label_name="label",data_type="train"):
        label_name_split = label_name.split(",")
        if(len(label_name_split) == 2):
            label_name_split.append('')

        if(len(label_name_split) <= 1):
            return -1
        labels_table = pd.DataFrame(pd.read_csv(label_address, header=0,error_bad_lines=False))
        patient_label_info = labels_table.loc[ (labels_table[label_name_split[2]] == data_type ),(label_name_split[0],label_name_split[1])]
        if(len(patient_label_info[label_name_split[1]].values) == 0):
            return -1
        else:
            #print("patient_label_info:",list(patient_label_info[label_name_split[0]]))
            
            #patient_label_dic = patient_label_info.set_index(label_name_split[0])[label_name_split[1]].to_dict()
            #return list(patient_ID_label_list2)
            return patient_label_info[label_name_split[0]],patient_label_info[label_name_split[1]]
def get_label(image_name, label_address,label_name='label',ID_prefix_num=15, data_type="train"):
        #print("label_name",label_name)
        label_name_split = label_name.split(",")
        if(len(label_name_split) == 2):
            label_name_split.append('')
       
        if(len(label_name_split) <= 1):
            return -1

        #print("label_name_split---0----",label_name_split[0],"----1----",label_name_split[1],"---2---",label_name_split[2],"----")
        #labels_table = pd.DataFrame(pd.read_csv(label_address,names=[label_name_split[0], label_name_split[1]], header=0))
        #print("label_address:",label_address)
        labels_table = pd.DataFrame(pd.read_csv(label_address, header=0,error_bad_lines=False))
        #print("=============image_name[0:15]:",image_name[0:15]) #df_sel2=df[df['day'].isin(['fri','mon'])]
        #patient_label_info = labels_table.loc[((labels_table[label_name_split[0]].str.contains(image_name[0:15]))& (labels_table[label_name_split[2]] != 'bak')), [label_name_split[1]]]
        patient_label_info = labels_table.loc[(((labels_table[label_name_split[0]].apply(lambda x: str(x))).str.contains(image_name[0:ID_prefix_num])) & (labels_table[label_name_split[2]] == data_type )), [label_name_split[1]]]
        if(len(patient_label_info[label_name_split[1]].values) == 0):
                return -1
        else:
                #print("patient_label_info.shape:",patient_label_info.shape)
                #print("patient_label_info:", patient_label_info[label_name])
                #print("image_name[0:",str(ID_prefix_num),"]:",image_name[0:ID_prefix_num]," label: ",int(patient_label_info[label_name_split[1]].values[0]))
                return int(patient_label_info[label_name_split[1]].values[0])


def get_feature_list(image_name, label_address,other_feature_name_strings='f1,f2',ID_prefix_num=15):
        other_feature_name_strings_split = other_feature_name_strings.split(",")
        if(len(other_feature_name_strings_split) <= 1):
            return []

        #labels_table = pd.DataFrame(pd.read_csv(label_address,names=[label_name_split[0], label_name_split[1]], header=0))
        labels_table = pd.DataFrame(pd.read_csv(label_address, header=0))
        #print(labels_table)
        #print("=============image_name[0:ID_prefix_num]:",image_name[0:ID_prefix_num]) #df_sel2=df[df['day'].isin(['fri','mon'])]
        #patient_label_info = labels_table.loc[((labels_table[label_name_split[0]].str.contains(image_name[0:15]))& (labels_table[label_name_split[2]] != 'bak')), [label_name_split[1]]]
        patient_other_feature_info = labels_table.loc[(((labels_table[other_feature_name_strings_split[0]].apply(lambda x: str(x))).str.contains(image_name[0:ID_prefix_num]))), other_feature_name_strings_split[1:]]
        if(len(patient_other_feature_info) == 0):   ####需要调试
                return []
        else:
                #print("------------------list(patient_other_feature_info.values[0].flatten()):",list(patient_other_feature_info.values[0].flatten()))
                return list(patient_other_feature_info.values[0].flatten())



#def HE_process(image_dir_root, label_address, size_square, label_name, image_num, ID_prefix_num, scan_window_suffix='*.png', data_type="train"):
def HE_process(**extra):
        
        image_dir_root = extra["images_dir"]
        label_address = extra["label_address"]
        size_square = extra["size_square"]
        label_name = extra["label_name"]
        image_num =  extra["image_num"]
        ID_prefix_num =  extra["ID_prefix_num"]
        scan_window_suffix =  extra["scan_window_suffix"]
        other_feature_name_strings = extra["other_feature_name_strings"]
        data_type=extra["data_type"]
        image_address_list = []
        image_name_list = []
        image_size1 = size_square
        image_size2 = size_square
        image_size3 = 3
        label_types = 2
        image_sum = 0
        image_data_list = []
        label_list = []
        other_feature_list_list = []
        image_dir_list = glob(join(image_dir_root, r'*/'))
        #print("svs_dirs:",image_dir_list)
        patient_ID_list, patient_label_list = get_patient_ID_label_list(label_address,label_name,data_type)
        patient_ID_label_list2 =list(zip(patient_ID_list,patient_label_list))
        #print("before table_patient_ID_label_list2",patient_ID_label_list2)
        for patient_ID_label in patient_ID_label_list2:
            print("========================patient_ID_label[0]:",patient_ID_label[0])
            #print("patient_ID_label[1]:",patient_ID_label[1])
            for image_dir in image_dir_list :
                    #print("image_dir:",image_dir)
                    image_num1 = image_num
                    image_address = glob(join(image_dir, scan_window_suffix))
                    len_addr = len(image_address) - 1
                    if(len(image_address) == 0):
                            #print("#####################image_address:", image_address)
                            continue
                    if len_addr < image_num1:
                            image_num1 = len_addr
                    #src_list = [random.randint(0, len_addr) for i in range(image_num1)]
                    src_list = list(range(0, image_num1))
                    #print(src_list)
                    for i in range(image_num1):
                            if (len(image_address) < i + 1):
                                break
                            img_num = src_list[i]
                            image_address_split = image_address[img_num].split("/")
                            #print(image_address[img_num])
                            image_name = image_address_split[-1]
                            image_name_split = image_name.split("_")
                            #print(image_name_split)
                            svs_name = image_name_split[0]
                            #print("len  of patient_ID_label[0]:",len(str(patient_ID_label[0]))," len of svs_name:", len(svs_name[0:int(ID_prefix_num)]))
                            if str(patient_ID_label[0]) != svs_name[0:int(ID_prefix_num)]:
                                break
                            #print("svs_name[0:",str(ID_prefix_num),"]:",svs_name[0:int(ID_prefix_num)])
                            #if(patient_ID_label[0] in svs_name[0,ID_prefix_num])
                            patient_label = -1
                            #patient_label = random.randint(0,1)
                            patient_other_feature_list = []
                            #print("label_address:",label_address)
                            #print("svs_name",svs_name)
                            patient_label = get_label(svs_name,label_address,label_name, ID_prefix_num,data_type)
                            if (patient_label == -1):
                                continue
                            patient_other_feature_list = get_feature_list(image_name, label_address, other_feature_name_strings, ID_prefix_num)
                            patient_other_feature_list = list(map(float, patient_other_feature_list))
                            #print("svs_name:",svs_name,"______________patient_label: ",patient_label)
                            #print("**************patient_other_feature_list:",patient_other_feature_list)
                            ##
                            #if need_save_WGI:
                            #       #https://pypi.org/project/pyvips/2.0.4/
                            #       #img = pyvips.Image.new_from_file(image_address[0], access='sequential')
                            #       #WGI_address = image_address[0] + ".tiff"
                            #       #img.write_to_file(WGI_address)

                            svs_file_slide = openslide.open_slide(image_address[i])
                            image_sum += 1
                            #print("image_address[0]:",image_address[i])
                            #print("svs_file_slide:", svs_file_slide.shape)
                            need_deepzoom = False

                            #####choose region
                            #svs_address = image_address[i]
                            #region_list = choose_region(svs_file_slide, size_square, svs_address)
                            #image_data = np.array(svs_file_slide.read_region((0, 0), 0, (size_square,size_square)).convert('RGB'))
                            image_data = np.array(svs_file_slide.read_region((0, 0), 0, (size_square,size_square)))
                            #####choose region
                            ##benben
                            image_data = image_data[:,:,0:3]
                            ##benben
                            image_address_list.append(image_address)
                            image_name_list.append(image_name)
                            label_list.append(patient_label)
                            image_data_list.append(image_data)
                            other_feature_list_list.append(patient_other_feature_list)


                            #for svs_file in  glob(join(svs_dir, '*.svs')):
        #get_label_list(image_name_list, label_address,label_list)
        #print("svs_file_list:", image_name_list)
        #print(len(image_data_list))
        return label_list, image_data_list, image_name_list, other_feature_list_list

def prepare_features_and_labels(x,y):
    x = tf.cast(x,tf.float32)/255.0
    y = tf.cast(y,tf.int64)
    return x,y

if __name__ == '__main__':

        print('!!!!!!!!!!!!!!! COMING HE_MODEL FUNCTION !!!!!!!!!!!!!!!!!')
        print("###########################################this is beginning: \n")
        parser = argparse.ArgumentParser(description='manual to this script', epilog="authors of this script are PengChao YeZixuan XiaoYupei and Ben ")
        #parser.add_argument('--gpus', type=str, default = None)
        #parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--images_dir', type=str, default = "../../data/TCGA/lung/")
        parser.add_argument('--label_address', type=str, default = "reg-tmb.csv")
        parser.add_argument('--save_model_address', type=str, default = "../../data/result/my_model.CNN_3")
        parser.add_argument('--train_model', type=str, default = "begin")
        parser.add_argument('--result_dir', type=str, default = "../../data/results/")
        parser.add_argument('--image_sizes_list', type=str, default = "512,512,3")
        #parser.add_argument('--size_square', type=int, default = 512)
        parser.add_argument('--label_types', type=int, default = 2)
        #parser.add_argument('--model_number', type=int, default=0,help="choose the model:0为CNN_3, 1为VGG_16, 2为resnet_34,3为resnet_50, 4为GoogleNet,5为Inception_V3, 6为Inception_V4,7为Inception_resnet_v1, 8为Inception_resnet_v2,9为ShuffleNet")
        parser.add_argument('--model', type=str, default="VGG16",help="choose the model:0为CNN_3, 1为VGG_16, 2为resnet_34,3为resnet_50, 4为GoogleNet,5为Inception_V3, 6为Inception_V4,7为Inception_resnet_v1, 8为Inception_resnet_v2,9为ShuffleNet")
        parser.add_argument('--parameters_dict', type=str, default="{1:1}", help="parameters_dict")
        parser.add_argument('--cross_validation_dict', type=str, default="{1:[1,2]}", help="parameters_dict")
        parser.add_argument('--epochs', type=int, default=10)
        parser.add_argument('--n_splits', type=int, default=2, help="Cross validation")
        parser.add_argument('--cross_validation_method', type=int, default=1, help="1:KF.split 2:cross_val_predict 3:cross_val_score")
        parser.add_argument('--data_type', type=str, default="train", help="train or test")
        parser.add_argument('--label_name', type=str, default="file_name,labels", help="label_name")
        parser.add_argument('--other_feature_name_strings', type=str, default="file_name,f1,f2", help="feature_list")
        parser.add_argument('--scan_window_suffix', type=str, default="*.png", help="scan_window_suffix")
        parser.add_argument('--image_num', type=int, default=1, help="The number of small images selected for each large image")
        parser.add_argument('--ID_prefix_num', type=int, default=15)
        parser.add_argument('--batch_size', type=int, default=2,help="batch_size")
        parser.add_argument('--tensorflow_version', type=str, default="2.0",help="batch_size")
        parser.add_argument('--activation_function', type=str, default="relu",help="activation default relu")
        parser.add_argument('--init_mode', type=str, default="uniform",help="init_mode default uniform")
        parser.add_argument('--optimizer', type=str, default="Adam",help="optimizer default Adam")
        parser.add_argument('--metrics_type', type=str, default="accuracy",help="--metrics_type default accuracy")
        parser.add_argument('--roc_address', type=str, default="roc.png",help="roc_address")
        parser.add_argument('--roc_title', type=str, default="ROC curve",help="roc curve title")

        args = parser.parse_args()
        print("args.label_address:",args.label_address)
        print("@@@@@@@@@@@@@@@@@@@args.images_dir:",args.images_dir)
        print("args.label_name:",args.label_name)
        image_sizes_list_from_args = args.image_sizes_list
        size_square = image_sizes_list_from_args.split(",")[0] 
        size_square = int(size_square) 
        print("parameters_dict::",args.parameters_dict)
        print("#############################################this is the end of argument! \n")

        model_types = ['CNN_3', 'VGG_16', 'resnet_34', 'resnet_50', 'GoogleNet', 'Inception_V3', 'Inception_V4',
                       'Inception_resnet_v1', 'Inception_resnet_v2','ShuffleNet']
        #save_model_address = args.save_model_address + model_types[args.model_number]
        #####################################
        other_feature_list_list = []
        image_name_list = []
        kwargs={
        "n_splits":args.n_splits,
        "save_model_address":args.save_model_address,
        "train_model":args.train_model,
        "model_types":args.model,
        "image_sizes_list":args.image_sizes_list,
        "label_types":args.label_types,
        "parameters_list":args.parameters_dict,
        "activation_function":args.activation_function,
        "init_mode":args.init_mode,
        "optimizer":args.optimizer,
        "metrics_type":args.metrics_type,
        "epochs":args.epochs,
        "batch_size":args.batch_size,
        "cross_validation_method":args.cross_validation_method,
        "roc_address":args.roc_address,
        "roc_title":args.roc_title,
        "other_feature_list_list":other_feature_list_list,
        "image_name_list":image_name_list,
        "data_type":args.data_type, 
        "images_dir":args.images_dir,
        "label_address":args.label_address,
        "size_square":size_square,
        "label_name":args.label_name,
        "other_feature_name_strings":args.other_feature_name_strings,
        "image_num":args.image_num,
        "ID_prefix_num":args.ID_prefix_num,
        "scan_window_suffix":args.scan_window_suffix,
        "result_dir":args.result_dir
        }

        #####################################
        label_list, image_data_list, image_name_list,other_feature_list_list = HE_process(**kwargs)
        if(len(label_list) == 0):
            print("there are no images need to process")
            sys.exit(0) 
        kwargs["other_feature_list_list"] = other_feature_list_list
        kwargs["image_name_list"] = image_name_list
        image_data_list = np.reshape(image_data_list, (len(image_data_list), size_square, size_square, 3))

        image_data_list = image_data_list/255.0
        image_data_list = np.array(image_data_list)
        label_list = np.array(label_list)

        print("label_list before:",label_list)
        label_list = to_categorical(label_list, num_classes=None)
        #model_types[args.model_number] 
        print("----------------save_model_address:",args.save_model_address)
        print("----------------cross_validation_method", args.cross_validation_method)
        print("label_list:",label_list)
        if(args.tensorflow_version=="2.0"):
            if(args.data_type=="train"):
                build_model.model_train(X_train=image_data_list, Y_train=label_list, **kwargs)

             
            elif(args.data_type=="test"):
                print("-------------------enter mode test---------------------")
                #test_labels1, y_pred1 =build_model.model_test_multi(args.save_model_address, args.model, image_data_list, label_list,args.roc_address,args.roc_title)
                dataframe1 = build_model.model_test_multi(image_data_list=image_data_list, label_list=label_list, **kwargs)
                   
                #dataframe1.to_csv(args.result_dir + "test_result.csv",index=False,sep=',')
                #combination_df = [["patient_ID","my_pred","labels"]]
                #print("kwargs[ID_prefix_num]:",kwargs["ID_prefix_num"])
                #dataframe1["patient_ID"] = [x[0:int(kwargs["ID_prefix_num"])] for x in dataframe1['image_name_list']]
                #for image,df_for_this_image in dataframe1.groupby('patient_ID',as_index=True):
                #     print(df_for_this_image)
                #     temp_list=[image,df_for_this_image['y_pred'].mean(),df_for_this_image["test_labels"].min()]
                #     combination_df.append(temp_list)
                ####     df_for_this_image.sort_values('y_pred',inplace=True, ascending=False)
                ####     print(df_for_this_image.iloc[:10])
                #pd.DataFrame(combination_df).to_csv(args.result_dir + "test_result_combination.csv",index=False,sep=',')



        #tensorflow_CNN.model_train(train_images, train_labels, test_images, test_labels, args.size_square, args.size_square, 4, args.label_types)
        # liangyuebin
        # draw_heatmap.draw_heatmap(args.images_dir, args.labels_address, args.need_save_WGI, args.size_square)


        ###########################################
        #tensor方法
        # data = np.reshape(image_data_list, (len(image_data_list), args.size_square, args.size_square, 4))
        #
        # y = tf.one_hot(label_list, depth=1)
        # ds = tf.data.Dataset.from_tensor_slices((data, y))
        # ds = ds.map(prepare_features_and_labels)
        # ds = ds.shuffle(10000).batch(10)
        #
        # sample = next(iter(ds))
        #
        # train_images= sample[0]
        # test_images = sample[0]
        # train_labels= sample[1]
        # test_labels = sample[1]
        # CNN.model_train(save_model_address, model_types[args.model_number], train_images, train_labels, test_images,
        #                 test_labels, args.size_square, args.size_square, 4, args.label_types, args.epochs, args.times,
        #                 args.L1, args.L2, args.F1, args.F2, args.F3, args.n_splits)

