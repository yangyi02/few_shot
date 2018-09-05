CUR_DIR=$(pwd)

MODEL=base
EXP_NAME=${MODEL}_fs
MODEL_PATH=models/$EXP_NAME.pth
TENSORBOARD_PATH=tensorboard/$EXP_NAME
LOG_PATH=logs/$EXP_NAME.log
FIG_PATH=figures/$EXP_NAME
BATCH_SIZE=16
IMAGE_HEIGHT=128
IMAGE_WIDTH=384
OUTPUT_HEIGHT=16
OUTPUT_WIDTH=48
NUM_SCALE=1
NUM_CLASS=1

cd ..

# Train
# CUDA_VISIBLE_DEVICES=1 python main.py --train --exp_name=$EXP_NAME --data=kitti --model=$MODEL --batch_size=$BATCH_SIZE --image_height=$IMAGE_HEIGHT --image_width=$IMAGE_WIDTH --output_height=$OUTPUT_HEIGHT --output_width=$OUTPUT_WIDTH --num_scale=$NUM_SCALE --num_class=$NUM_CLASS --train_proportion=0.1 --train_iteration=300 --show_statistics=basic --save_model_path=$MODEL_PATH --tensorboard_path=$TENSORBOARD_PATH |& tee $LOG_PATH

# Test
# CUDA_VISIBLE_DEVICES=1 python main.py --test --exp_name=$EXP_NAME --data=kitti --model=$MODEL --batch_size=$BATCH_SIZE --image_height=$IMAGE_HEIGHT --image_width=$IMAGE_WIDTH --output_height=$OUTPUT_HEIGHT --output_width=$OUTPUT_WIDTH --num_scale=$NUM_SCALE --num_class=$NUM_CLASS --show_statistics=basic --init_model_path=$MODEL_PATH |& tee -a $LOG_PATH

# Visualize one image
IMAGE_NAME='/media/yi/DATA/data-orig/kitti/training/image_2/000003.png'
DEPTH_NAME='/media/yi/DATA/data-orig/kitti/training/disp_unsup/000003.png'
FLOW_NAME='/media/yi/DATA/data-orig/kitti/training/flow_unsup/000003.png'
BOX_NAME='/media/yi/DATA/data-orig/kitti/training/label_2/000003.txt'
CUDA_VISIBLE_DEVICES=1 python main.py --visualize --exp_name=$EXP_NAME --data=kitti --model=$MODEL --image_height=$IMAGE_HEIGHT --image_width=$IMAGE_WIDTH --output_height=$OUTPUT_HEIGHT --output_width=$OUTPUT_WIDTH --num_scale=$NUM_SCALE --num_class=$NUM_CLASS --init_model_path=$MODEL_PATH --image_name=$IMAGE_NAME --depth_name=$DEPTH_NAME --flow_name=$FLOW_NAME --box_name=$BOX_NAME --figure_path=$FIG_PATH

# Visualize a set of images
CUDA_VISIBLE_DEVICES=1 python main.py --visualize_all --exp_name=$EXP_NAME --data=kitti --model=$MODEL --image_height=$IMAGE_HEIGHT --image_width=$IMAGE_WIDTH --output_height=$OUTPUT_HEIGHT --output_width=$OUTPUT_WIDTH --num_scale=$NUM_SCALE --num_class=$NUM_CLASS --init_model_path=$MODEL_PATH --image_list=images.txt --figure_path=$FIG_PATH

cd $CUR_DIR
