#!/bin/bash
# $1  
# $2 cut	color	train	test 
# $3 
#  sh  root_dir cut image_suffix 
# sh  /data2/ben/data/zhongzhong/breast ndpi

# HE_process.sh /data2/ben/data/TCGA/breast  train csv&patient,relapse,sample_type&patient,age,tumor_stage 
#######################################################part 0

current_time=`date +"%Y_%m_%d_%H_%M_%S"`
echo $current_time
#current_time="2020_03_08_02_10_43"
project_dir=$1
images_dir="${project_dir}/images/"
result_dir="${project_dir}/results/"

###########################part 0


########################################part 1 start 
##step 1 prepare tumor ragion data
if [[ $2 =~ "cut" ]];
then
echo "in cut process"
echo $images_dir
echo $image_suffix
#image_suffix="ndpi"
image_suffix=$3
python data_prepare.py  \
--images_dir_root $images_dir \
--size_square 512 \
--prepare_types 1 \
--image_suffix  $image_suffix
fi

## step 2 color nomalization
if [[ $2 =~ "color" ]];
then
python Stain_Color_Normalization.py \
--log_dir "/data2/ben/HE/data/result/color_logs/" \
--data_dir $images_dir \
--tmpl_dir "/data2/ben/HE/data/result/color_tmp/" 
fi

###########################################part 1 end


##########################################part 2 construct model start

#size_square=512
image_sizes_list="512,512,3"
#model_type="resnet34"
#model_type="Xception"
#model_type="AlexNet"
#model_type="VGG16"
#parameters_dict="{'epochs':2,'batch_size':2,'activation':'relu','init_mode':'uniform','optimizer':'Adam'}"
#model_type="CNN3"
#model_type="AlexNet"
#model_type="InceptionV3"
#parameters_dict="{'reg':0.0002},{'reg':0.0001}"
activation_function="softmax"  ###'tanh' ## 'softmax'  'relu'

epochs=5  #temp
batch_size=2
times=3 #
tensorflow_version=2.0
roc_address=${save_model_address}"_"${mode}"_roc.pdf"
echo "----------------------"
echo $images_dir
#####construct model


function HE_model(){ 
##########

##########
python HE_model.py \
--images_dir $images_dir \
--save_model_address $save_model_address  \
--train_model $train_model \
--result_dir $result_dir \
--image_num $image_num_selected_in_single_dir \
--cross_validation_method 0 \
--model $model_type \
--data_type $data_type \
--parameters_dict $parameters_dict \
--activation_function $activation_function \
--label_name $label_name \
--other_feature_name_strings $other_feature_name_strings \
--scan_window_suffix "*CN.png"   \
--label_address $label_address  \
--n_splits 2  \
--label_types 2 \
--image_sizes_list $image_sizes_list \
--epochs $epochs \
--ID_prefix_num $ID_prefix_num \
--batch_size $batch_size \
--tensorflow_version $tensorflow_version \
--roc_address $roc_address  \
--roc_title "ROC curve"
#--scan_window_suffix "*.orig.png"
}

if [[ $2 =~ "train" ]] || [[ $2 =~ "test" ]] ;then

information=$3
#echo $information
array=(${information//+/ })

model_type=${array[0]}
save_model_address=${array[1]}
parameters_dict=${array[2]}
label_address=${project_dir}"/labels/"${array[3]}
label_name=${array[4]}
other_feature_name_strings=${array[5]}
image_num_selected_in_single_dir=${array[6]}
ID_prefix_num=${array[7]}
train_model=${array[8]}
echo "label_address"
echo $label_address


if [[ $2 =~ "train" ]];then
data_type="train"
HE_model
fi

if [[ $2 =~ "test" ]];then
data_type="test"
HE_model
fi

fi



############################################construct model end

#########################################draw heatmap start
if false;then

pattern="single"
step_x=100
step_y=100
begin_x=43000
begin_y=49000
dimensions_x=10000
dimensions_y=5000
image_address=../../data/TCGA/lung/images/01184c1e-c768-4459-a8ea-a443d18880d8/TCGA-50-5939-01Z-00-DX1.745D7503-0744-46B1-BC89-EBB8FCE2D55C.svs
elif false;then

pattern="single"
step_x=100
step_y=100
begin_x=0
begin_y=0
dimensions_x=-1
dimensions_y=-1
#image_address=../../data/geneis/lung/79807-14604-40x001.png
#image_address ../../data/TCGA/lung/01184c1e-c768-4459-a8ea-a443d18880d8/TCGA-50-5939-01Z-00-DX1.745D7503-0744-46B1-BC89-EBB8FCE2D55C.svs
##image_address /data2/ben/HE/data/xiehe/cesc/HE_image/S698161/S698161-A1.ndpi save_model_address="../../data/result/my_model.CNN_3test"
image_address="${project_dir}/images/79807-14604-40x002.png"

elif false;then

pattern="single"
step_x=500
step_y=500
begin_x=10000
begin_y=10000
dimensions_x=10000
dimensions_y=5000
image_address="${project_dir}/images/ed6eea33-2777-4e2a-83b1-fcbeed49ac90/TCGA-49-4510-11A-01-TS1.7310b502-a637-4912-857b-3c52214ad706.svs"
fi


if false;then
python draw_heatmap.py --step_x $step_x --step_y $step_y \
--save_model_address $save_model_address \
--pattern $pattern   \
--begin_x $begin_x \
--begin_y $begin_y \
--dimensions_x  $dimensions_x \
--dimensions_y $dimensions_y \
--image_address $image_address 

elif false;then  ###multiple

python draw_heatmap.py --step_x 500 --step_y 500 \
--save_model_address $save_model_address \
--pattern "multiple"   \
--scan_window_suffix "*.ndpi" \
--images_dir_root "../../data/xiehe/cesc/HE_image" \
--labels_address "../../data/xiehe/cesc/labels/xiehe.cesc.TMB.class.csv" \
--header_name "Samples,TMB"   \
--ID_prefix_num 7
fi

################################heatmap end
###./HE_process.sh /data/ben/data/TCGA/breast/ breast_relapse.type.csv test
