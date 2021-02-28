




#######################################################part 0

current_time=`date +"%Y_%m_%d_%H_%M_%S"`
echo $current_time
#current_time="2020_03_08_02_10_43"
#my_model.CNN_3_reg-metastasis20200304
#images_dir_root="/data2/ben/HE/data/TCGA/lung/"
#images_dir_root="/data2/ben/HE/data/xiehe/cesc/HE_image"
project_dir="/data2/ben/data/TCGA/cesc/"
images_dir="${project_dir}/images/"


if false;then
labels_address="${project_dir}/labels/labels_immune1.csv"
label_name="patients,POLE"
#label_name="patients,POLD1"
#label_name="patients,PBRM1"
#label_name="patients,STK11"
#label_name="patients,DNMT3A"
#label_name="patients,EGFR"
#label_name="patients,TP53"
#label_name="patients,KRAS"
#label_name="patients,CD274"

save_model_address="${project_dir}/results/"${current_time}"my_model.immune1"

elif false;then
labels_address="${project_dir}/labels/labels_target1.csv"
label_name="patients,AKT1"
#label_name="patients,ALK"
#label_name="patients,BRAF"
#EGFR,
#label_name="patients,FGFR1"
#label_name="patients,FGFR2"
#label_name="patients,HRAS"
#,KRAS,
#label_name="patients,MET"
#label_name="patients,NRAS"
#label_name="patients,NTRK1"
#label_name="patients,NTRK2"
#label_name="patients,NTRK3"
#label_name="patients,PIK3CA"
#label_name="patients,RET"
#label_name="patients,ROS1"
#label_name="patients,TSC1"
save_model_address="${project_dir}/results/"${current_time}"my_model.target1"

elif false;then ##this is metastasis

labels_address="${project_dir}/labels/met.col1_col3_train.csv"
label_name="he_name,metastasis"
save_model_address="${project_dir}/results/"${current_time}"my_model.CNN_3_reg-metastasis"
############save_model_address="../../data/result/my_model.CNN_3_20191210--" 

elif false;then #this is immune

labels_address="${project_dir}/labels/luad_immune.csv"
label_name="patients,POLE"
save_model_address="${project_dir}/results/"${current_time}"my_model.CNN_3_reg-luad_immune"

elif true;then ##this is TMB

labels_address="${project_dir}/labels/reg_tmb.csv"
label_name="file_name,labels"
#temp  save_model_address="${project_dir}/results/"${current_time}"my_model.tmb"
save_model_address="${project_dir}/results/my_model.CNN_3_reg-metastasis20200304"
############save_model_address="../../data/result/my_model.CNN_3_20191210--" 

fi


###########################part 0










###############part 1 start 
##step 1 prepare tumor ragion data
if false; then
image_suffix="*.svs"
python data_prepare.py  \
--images_dir_root $images_dir \
--size_square 512 \
--prepare_types 1 \
--image_suffix  $image_suffix
fi

## step 2 color nomalization
if false; then
python Stain_Color_Normalization.py \
--log_dir "/data2/ben/HE/data/result/color_logs/" \
--data_dir "/data2/ben/HE/data/TCGA/lung/" \
--tmpl_dir "/data2/ben/HE/data/result/color_tmp/" 
fi


###########################################part 1 end



##########################################part 2 construct model start




size_square=512
epochs=15  #temp
batch_size=2
times=3 #
L1=10 #"the number of the first conv2d
F1=5 # "the size of the first conv2d layer")

F2=2 #"the size of the first maxpooling layer")

L2=10 #"the number of the second conv2d ")
F3=3 #"the size of the second conv2d layer"
tensorflow_version=2.0
roc_address=${save_model_address}"_"${mode}"_roc.pdf"
#####construct model
if false;then
python HE_model.py --save_model_address $save_model_address  \
--image_num 1 \
--cross_validation_method 3 \
--label_name $label_name \
--scan_window_suffix *color_norma.png   \
--labels_address $labels_address  \
--n_splits 2  \
--label_types 2 \
--images_dir $images_dir \
--size_square $size_square \
--epochs $epochs \
--times $times \
--L1 $L1 \
--L2 $L2 \
--F1 $F1 \
--F2 $F2 \
--F3 $F3 \
--batch_size $batch_size \
--tensorflow_version $tensorflow_version \
--roc_address $roc_address  \
--roc_title "ROC curve"
#--scan_window_suffix "*.orig.png"
fi
############################################construct model end

#########################################draw heatmap start
if true;then

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


if true;then
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
