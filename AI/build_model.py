from __future__ import absolute_import, division, print_function, unicode_literals
#from skimage import io, transform
import glob
import os
import sys
import numpy as np
import time
#import xlrd
#from openpyxl import load_workbook
#from openpyxl import Workbook
#from openpyxl.writer.excel import ExcelWriter
import cv2
# import xlwt
import tensorflow as tf
import pandas
from keras import losses
# from tensorflow.keras import layers
import tensorflow.keras as keras
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from scipy import interp
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score
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
###
sys.path.append("..")
#print('the path of common is {}'.format(sys.path))
import COMMON.draw_roc as draw_roc
#from AI.models.resnet import resnet34
#from AI.models.ception import Xception
#from AI.models.models import LeNet
#from AI.models.models import CNN3
#from AI.models.models import AlexNet
#from AI.models.models import AlexNet1

'''
  Add in 20200826
'''
#from AI.models.models import VGG16
#from AI.models.models import VGG19
#from AI.models.models import InceptionV3

'''
    Add ImageNet-based transfer learning in 20200903
    Selected the best model according to RandomizedSearchCv in 20201105
    name: xisx
'''
import itertools
from sklearn.model_selection import RandomizedSearchCV

'''
   add from AI.models.models import CNN_features_model
   name: ben

'''
#from AI.models import CNN_features_model 
import AI.models.CNN_features_model as CNN_features_model
import AI.models.VGG16 as VGG16
import AI.models.VGG19 as VGG19
import AI.models.ResNet as ResNet
#def create_model(model_types, image_sizes_list,label_types, parameters_list=None, activation='relu', init_mode='uniform', optimizer='Adam', metrics_type='accuracy', compile_loss=losses.categorical_crossentropy,**kwargs):
def create_model( **kwargs):

    model_types = kwargs["model_types"] 
    #parameters_list = kwargs["parameters_list"] 
    #parameters_dict = eval(parameters_list)
    #print("parameters_dict:",parameters_dict)
    #optimizer = parameters_dict["optimizer"]
    #optimizer = kwargs["optimizer"]
    #loss = parameters_dict["loss"]
    #loss = kwargs["loss"]
    #metrics_type = parameters_dict["metrics_type"]
    #learning_rate = float(parameters_dict["learning_rate"])
    #metrics_type = kwargs["metrics_type"]
    module_name = ''.join([model_types,".",model_types])
    #model = eval(model_types)(image_sizes_list, label_types, parameters_list, activation, init_mode, **kwargs)
    #model = eval(model_types)(**kwargs)
    model = eval(module_name)(**kwargs)

    # def get_lr_metric(optimizer):
    #     def lr(y_true, y_pred):
    #         return optimizer.lr
    #
    #     return lr
    #
    # lr_metric = get_lr_metric(optimizer)
    #loss='sparse_categorical_crossentropy'
    #optimizer = keras.optimizers.SGD(learning_rate=0.5)
    #optimizer = keras.optimizers.SGD(lr=0.02, decay=0.9, momentum=0.9, nesterov=True)  

    #model.compile(loss=loss, optimizer=optimizer, metrics=[metrics_type])
    #model.summary()

    return model


#def train_by_all(n_splits, save_model_address, model_types, train_images, train_labels,test_images,test_labels, image_sizes_list, label_types, parameters_list, activation='relu', init_mode='uniform', optimizer='Adam', metrics_type='accuracy',epochs =2, batch_size=2,roc_address="roc.pdf",roc_title="ROC curve"):
def train_by_all(X_train, Y_train, **kwargs):
    #epochs = kwargs["epochs"]
    #model = tf.keras.models.load_model(kwargs["save_model_address"]) 
    print("this is after load model")
    result_dir = kwargs["result_dir"]
    print("parameters_list:",kwargs["parameters_list"])
    parameters_dict = eval(kwargs["parameters_list"])
    epochs=parameters_dict["epochs"]
    batch_size= parameters_dict["batch_size"]
    #batch_size = kwargs["batch_size"]
    #parameters_list1 = parameters_list.split(",")
    #parameters_dic = eval(kwargs["parameters_list"])
    model_types = kwargs["model_types"]
    i = 0
    print("-------------------------in the beginning of train_by_all_trainning_set:")
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    if True:
        # 建立模型，并对训练集进行测试，求出预测得分
        # 划分训练集和测试集
        #X_train = np.array(train_images)
        #Y_train = np.array(train_labels)
        #Y_train = to_categorical(Y_train, num_classes=None)
        #X_test = X_train
        #Y_test = Y_train
        
        # 建立模型(模型已经定义)
        print("|||||||||||||||before eval:")
        #model = create_model(model_types, image_sizes_list,label_types, parameters_list1[0], activation, init_mode, optimizer, metrics_type)
        print("@@@@@@@@@@@@@@train_model:", kwargs["train_model"])
        if(kwargs["train_model"]=="load"):
            model = tf.keras.models.load_model(kwargs["save_model_address"]) 
        else:
            model = create_model(X_train=X_train, Y_train=Y_train, **kwargs)

        # 训练模型
        print("Y_train:",Y_train)
        #eval(model_types).fit(X_train, Y_train, **kwargs)
        eval(model_types).fit(X_train=X_train, Y_train=Y_train, model=model, **kwargs)
        #probas_ = model.predict(X_test, batch_size=batch_size)
        #now change to common draw_roc_curve(Y_test[:,1], probas_[:, 1],roc_address,roc_title=roc_title)
        auc_title = ""
        #draw_roc.draw_roc_curve(Y_test[:,1], probas_[:, 1],kwargs["roc_address"],auc_title,kwargs["roc_title=roc_title"])
        #draw_roc.draw_roc_curve(Y_test, probas_, kwargs["roc_address"], auc_title, kwargs["roc_title"])
    
    print("save_model_address", kwargs["save_model_address"])
    model.save(kwargs["save_model_address"])  # creates a HDF5 file 'my_model.h5'
    print(" model has been saved to ", kwargs["save_model_address"])
    # step 7
    #del model  # deletes the existing model
    #print("-------------after del model")

    #restored_model = tf.keras.models.load_model(kwargs["save_model_address"]) 
    #dataframe1 = eval(model_types).predict(model=restored_model,image_data_list=X_train,label_list=Y_train,**kwargs) 
    # step 7
    dataframe1 = eval(model_types).predict(model=model,image_data_list=X_train,label_list=Y_train,**kwargs) 
    print("result of pred:")
     
    print(dataframe1)

def cross_validation_one_time(n_splits, save_model_address, model_types, train_images, train_labels, image_sizes_list, label_types, parameters_list, activation, init_mode, optimizer, metrics_type, epochs, batch_size, roc_address, roc_title, **extra):
#def cross_validation_one_time(n_splits, save_model_address, model_types, train_images, train_labels, image_sizes_list, label_types, parameters_list, activation, init_mode, optimizer, metrics_type, roc_address, roc_title):
    if n_splits == 1:
        TF_split_list_2 = [(np.array( list(range(len(train_labels))) ), np.array( list(range(len(train_labels))) ))]
    else:
        KF = KFold(n_splits=n_splits, shuffle=True, random_state=7)
        TF_split_list_2 = KF.split(train_images)
    i = 0

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    print("label_types:",label_types)
    model = KerasClassifier(build_fn=create_model,model_types=model_types, image_sizes_list=image_sizes_list, label_types=label_types, parameters_list=parameters_list, activation=activation, init_mode=init_mode, optimizer=optimizer, metrics_type=metrics_type)
    print("roc_address: ",roc_address)
    pdf = PdfPages(roc_address)         #先创建一个pdf文件
    plt.figure
    for train_index, test_index in TF_split_list_2: ##KF.split(train_images):
        # 建立模型，并对训练集进行测试，求出预测得分
        # 划分训练集和测试集
        X_train, X_test = np.array(train_images)[train_index], np.array(train_images)[test_index]
        Y_train, Y_test = np.array(train_labels)[train_index], np.array(train_labels)[test_index]

        print("Y_train:",Y_train)
        print("n_splits:",n_splits)

        early_stopping = keras.callbacks.EarlyStopping(monitor='loss',min_delta=0.001, patience=5, verbose=2)
        model.fit(X_train, Y_train, batch_size=batch_size, validation_data=(X_test, Y_test), epochs=epochs, verbose=2,  shuffle=False, callbacks=[early_stopping])
        probas_ = model.predict(X_test, batch_size=batch_size)
        ##################
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(Y_test[:,1], probas_)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        i += 1

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(roc_title)
    plt.legend(loc="lower right")
    plt.savefig(roc_address)
    plt.show()
    pdf.savefig()                            #将图片保存在pdf文件中
    plt.close()
    pdf.close()
    print("aucs:",aucs)
    print("mean_auc",mean_auc, " ",std_auc)

    # step7
    print("-------------after del model")
    return mean_auc,model

def cross_validation_select_parameters_by_hands(n_splits, save_model_address, model_types, train_images, train_labels, test_images, test_labels, image_sizes_list, label_types, parameters_list, activation, init_mode, optimizer, metrics_type, epochs, batch_size, roc_address, roc_title, **extra):
    print("parameters_list:",parameters_list)
    parameters_dic = eval(parameters_list)
    all_parameters_name = sorted(parameters_dic)
    combination_parameters = itertools.product(*(parameters_dic[name] for name in all_parameters_name))
    y_scores_max = -1
    times_best = 0
    X_train = np.array(train_images)
    Y_train = np.array(train_labels)
    Y_train = to_categorical(Y_train, num_classes=None)
    print("-------------------------- roc_address is :", roc_address)
    times = 0

    for parameters_list_1 in combination_parameters:
        times += 1
        roc_address1 = roc_address + "_" +str(times) + ".pdf"
        #y_scores_mean,model = cross_validation_one_time(n_splits, save_model_address, model_types, X_train, Y_train, image_sizes_list, label_types, para, activation, init_mode, optimizer, metrics_type, roc_address1, roc_title)
        y_scores_mean,model = cross_validation_one_time(n_splits, save_model_address, model_types, X_train, Y_train, image_sizes_list, label_types, parameters_list_1, activation, init_mode, optimizer, metrics_type, epochs, batch_size, roc_address1, roc_title, **extra)
        if y_scores_mean > y_scores_max:
            combination_parameters_best = parameters_list_1
            y_scores_max = y_scores_mean
        print("y_scores_mean:", y_scores_mean, "  parameters_list_best:", combination_parameters_best)
    #### construst model by best hyperparameters
    if times == 1:
        #y_scores_mean,model = cross_validation_one_time(1, save_model_address, model_types, X_train, Y_train, image_sizes_list, label_types, combination_parameters_best, activation, init_mode, optimizer, metrics_type, epochs, batch_size, roc_address, roc_title)
        y_scores_mean,model = cross_validation_one_time(1, save_model_address, model_types, X_train, Y_train, image_sizes_list, label_types, combination_parameters_best, activation, init_mode, optimizer, metrics_type, epochs, batch_size, roc_address, roc_title, **extra)
    print("save_model_address", save_model_address)
    model.model.save(save_model_address)  # creates a HDF5 file 'my_model.h5'
    print(" model has been saved to ", save_model_address)


def Keras_Classifier(n_splits, save_model_address, model_types, train_images, train_labels, test_images, test_labels, image_sizes_list, label_types, parameters_list, activation='relu', init_mode='uniform', optimizer='Adam', metrics_type='accuracy', cross_validation_method=1, epochs=2, batch_size=2, roc_address="roc.pdf", roc_title="ROC curve"):
    print("begin of keras_classifier: image_sizes_list:",image_sizes_list, "type:", image_sizes_list)

    parameters_list1 = parameters_list.split(",")
    model = create_model(model_types=model_types, image_sizes_list=image_sizes_list, label_types=label_types, parameters_list=parameters_list1[0], activation=activation, init_mode=init_mode, optimizer=optimizer, metrics_type=metrics_type)
    kfold = KFold(n_splits=int(n_splits), shuffle=True, random_state=5000)
    for train, test in kfold.split(train_images):
        print("train:",train, "test",test)
    if False:
        scoring = ['precision_macro','recall_macro']
        scores = cross_validate(model, train_images, train_labels,scoring=scoring,cv=kfold,return_train_score=False)
        sorted(scores.keys())
        #['fit_time', 'score_time', 'test_precision_macro', 'test_recall_macro']
        print(scores['test_recall_macro'])
    elif False:
        if False:
            # sklearn.model_selection.cross_val_score Returns  scoresarray of float, shape=(len(list(cv)),)  Array of scores of the estimator for each run of the cross validation.
            y_pre = cross_val_score(model, train_images, train_labels, cv=kfold)
        else:
            y_pre = cross_val_predict(model, train_images, train_labels, cv=kfold)
        auc_title = ""
        draw_roc.draw_roc_curve(train_labels, y_pre, roc_address, auc_title, roc_title=roc_title)

    else:
        '''
        y_pre = cross_val_score(model, train_images, train_labels, cv=kfold)
        print("y_pre is ", y_pre)

        # predictions = pandas.Series(y_pre)
        # print("pre:",y_pre,"predictions:",predictions)
        auc_title = ""
        draw_roc.draw_roc_curve(train_labels, y_pre, roc_address, auc_title, roc_title=roc_title)
        '''
        pass


    #X_train, X_test = np.array(train_images), np.array(test_images)
    #Y_train, Y_test = np.array(train_labels), np.array(test_labels)
    #Y_train = to_categorical(Y_train, num_classes=None)
    #Y_test = Y_train
    # 建立模型(模型已经定义)
    #model.fit(X_train, Y_train, batch_size=batch_size, validation_data=(X_test, Y_test), epochs=epochs)
    #model.fit(train_images, train_labels, batch_size=2, validation_data=(train_images, train_labels), epochs=2)
    model.save(save_model_address)
    print(" this  is the end")


def algorithm_pipeline(X_train_data, y_train_data, X_test_data, y_test_data,
                       model, param_grid, cv=10, scoring_fit='neg_mean_squared_error',
                       do_probabilities=False, search_type=0):
    print("-------------------------param_grid is:", type(param_grid))

    if search_type == 0:
        gs = GridSearchCV(estimator=model,
                          param_grid=param_grid,
                          cv=cv,
                          verbose=2,
                          # n_jobs=-1,
                          scoring=scoring_fit)

    if search_type == 1:
        gs = RandomizedSearchCV(estimator=model,
                                param_distributions=param_grid,
                                cv=cv,
                                verbose=2,
                                n_jobs=1,
                                scoring=scoring_fit)

    Early_sp = tf.keras.callbacks.EarlyStopping(
        monitor='accuracy',
        patience=3,
        mode='auto',
        verbose=1
    )

    fitted_model = gs.fit(X_train_data, y_train_data, callbacks=[Early_sp])  # , callbacks=[Early_sp]

    best_model = gs.best_estimator_
    if do_probabilities:
      pred = fitted_model.predict_proba(X_test_data)
    else:
      pred = fitted_model.predict(X_test_data)
    print("pred:",pred)

    return best_model, pred, fitted_model


def cross_validation_select_parameters_by_GridSearchCV(n_splits, save_model_address,model_types,train_images,train_labels, test_images, test_labels, image_sizes_list, label_types, parameters_list,activation, init_mode, optimizer, metrics_type, epochs, batch_size, roc_address, roc_title):
    '''

    :param parameters_list:
         For example: {
                        'times': [2],
                        'epochs': [100,150,200],
                        'batch_size': [32, 128],
                        'optimizer': ['Adam', 'Nadam'],
                        'dropout_rate': [0.2, 0.3]
                        'activation': ['relu', 'elu']
                        'init_mode':['uniform', 'lecun_uniform', 'normal']
                     }
    :return: best model based on the parameters_list information
    '''
    print("begin of crosss_validation_select_parameters_by_GridSearchCV: ")
    y_scores_max = -1
    times_best = 0
    X_train = np.array(train_images)
    Y_train = np.array(train_labels)

    ##### split data start
    #seed = 7 #重现随机生成的训练
    #test_size = 0.33 #33%测试，67%训练
    #test_size = 0.50 #33%测试，67%训练
    #X_train, X_test, Y_train, Y_test = train_test_split(train_images, train_labels, test_size=test_size, random_state=seed)
    ##### split data end

    parameters_dic = eval(parameters_list)
    param_grid = parameters_dic

    #init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
    #init_mode = ['uniform','lecun_uniform','normal']
    #param_grid = dict(init_mode=init_mode)
    #model = XGBClassifier()
    #now model = KerasClassifier(build_fn = build_cnn, activation = 'relu', dropout_rate = 0.2, optimizer = 'Adam', fs1 = 5, times = times, init_mode='uniform', verbose=1)

    model = KerasClassifier(build_fn=create_model,model_types=model_types, image_sizes_list=image_sizes_list, label_types=label_types, parameters_list=parameters_list)
    #model = create_model(model_types=model_types, image_sizes_list=image_sizes_list, label_types=label_types, parameters_list=parameters_list)
    best_model, pred, fitted_model = algorithm_pipeline(train_images, train_labels, test_images, test_labels, model, param_grid, cv=n_splits, scoring_fit='neg_log_loss',do_probabilities=True)
    auc_title = ""
    draw_roc.draw_roc_curve(test_labels, pred[:,1], roc_address, auc_title, roc_title=roc_title)
    best_model.model.save(save_model_address)


def cross_validation_select_parameters_by_RandomizedSearchCV(n_splits, save_model_address, model_types,
                                                             train_images, train_labels, test_images, test_labels,
                                                             image_sizes_list, label_types, parameters_list, roc_address, roc_title):
    y_scores_max = -1
    times_best = 0
    X_train = np.array(train_images)
    Y_train = np.array(train_labels)

    parameters_dic = eval(parameters_list)
    param_grid = parameters_dic

    model = KerasClassifier(build_fn=create_model, model_types=model_types, image_sizes_list=image_sizes_list,
                            label_types=label_types, parameters_list=parameters_list)
    best_estimator, pred, fitted_model = algorithm_pipeline(train_images, train_labels, test_images, test_labels,
                                                            model, param_grid=param_grid, cv=n_splits,
                                                            scoring_fit='neg_log_loss', do_probabilities=True, search_type=1)

    auc_title = ""
    draw_roc.draw_roc_curve(test_labels, pred[:, 1], roc_address, auc_title, roc_title=roc_title)
    best_model.model.save(save_model_address)


#def model_train(n_splits, save_model_address, model_types, train_images, train_labels,test_images, test_labels, image_sizes_list, label_types, parameters_list, activation='relu', init_mode='uniform', optimizer='Adam', metrics_type='accuracy', epochs=2, batch_size=2, cross_validation_method=1, roc_address="roc.pdf", roc_title="ROC curve"):
def model_train( **kwargs):
    tf.config.threading.set_inter_op_parallelism_threads = 50
    # 只使用一个线程
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    cross_validation_method = kwargs["cross_validation_method"] 
    if cross_validation_method == 1:
        pass
    elif cross_validation_method == 2:  #### cross_validation 1 time on  cross_val_predict
        pass
        #Keras_Classifier(n_splits, save_model_address, model_types, train_images, train_labels, test_images, test_labels, image_sizes_list, label_types, parameters_list, activation, init_mode, optimizer, metrics_type, cross_validation_method, epochs, batch_size, roc_address, roc_title)
    elif cross_validation_method == 3:
        pass
        #cross_validation_select_parameters_by_hands(n_splits, save_model_address, model_types, train_images, train_labels, test_images, test_labels, image_sizes_list, label_types, parameters_list, activation, init_mode, optimizer, metrics_type, epochs, batch_size, roc_address, roc_title)
    elif cross_validation_method == 4:  ### cross_validation on hyperparameters  on GridSearchCV
        pass
        #cross_validation_select_parameters_by_GridSearchCV(n_splits, save_model_address, model_types, train_images, train_labels, test_images, test_labels, image_sizes_list, label_types, parameters_list, activation, init_mode, optimizer, metrics_type, epochs, batch_size, roc_address, roc_title)
    elif cross_validation_method == 5:
        pass
        #cross_validation_select_parameters_by_RandomizedSearchCV(n_splits, save_model_address, model_types, train_images, train_labels, test_images, test_labels, image_sizes_list, label_types, parameters_list, roc_address, roc_title)
    elif cross_validation_method == 0:
        train_by_all(**kwargs)


def model_test_multi(**kwargs):
    restored_model = tf.keras.models.load_model(kwargs["save_model_address"])

    #test_images = np.array(kwargs["image_data_list"])
    #test_labels = np.array(kwargs["label_list"])

    #test_labels = to_categorical(test_labels, num_classes=None)
    # step9
    #need loss, acc = restored_model.evaluate(test_images, test_labels)
    #need print("Restored model, accuracy:{:5.2f}%".format(100 * acc))
    # pred = restored_model.predict(test_images[:2])
    #print('predict:', pred )
    # https://blog.csdn.net/wwwlyj123321/article/details/94291992
    # 利用model.predict获取测试集的预测值
    #20201204benbak y_pred = restored_model.predict(test_images, batch_size=1)
    model_types = kwargs["model_types"]
    y_pred_df = eval(model_types).predict(model=restored_model,**kwargs)
    #draw_roc_curve(test_labels[:,1], y_pred[:,1], roc_address, roc_title)
    auc_title = ""
    ####下面的语句需要检查其中test_labels是否为二维数据。
    #draw_roc.draw_roc_curve(test_labels[:,1], y_pred[:, 1],roc_address,auc_title,roc_title=roc_title)
    #return test_labels[:,1], y_pred[:, 1]
    #temp draw_roc.draw_roc_curve(test_labels, y_pred,roc_address,auc_title,roc_title=roc_title)
    return  y_pred_df

def model_test_single(save_model_address, test_image):
    restored_model = tf.keras.models.load_model(save_model_address)
    test_image = np.array(test_image)
    # step9
    # pred = restored_model.predict(test_images[:2])
    #print('predict:', pred )
    # https://blog.csdn.net/wwwlyj123321/article/details/94291992
    # 利用model.predict获取测试集的预测值
    y_pred = restored_model.predict(test_image, batch_size=1)
    return y_pred





