#!/bin/bash
CUR_DIR=$(dirname "$0")
source $CUR_DIR/../set_env.sh

MODEL=base

# Mode can be test mode, fast training mode or slow training mode
# test mode test the shell script
# fast mode train the model with fewer data and fewer iterations
# full mode train the model with full data and full iterations
MODE=$1
if [ $MODE = 'test' ]; then
  EXP_NAME=${MODEL}_test
  TRAIN_PROPORTION=0.1
  TRAIN_ITERATION=2
  TEST_ITERATION=1
  TEST_INTERVAL=2
  TEST_PROPORTION=0.05
elif [ $MODE = 'fast' ]; then
  EXP_NAME=${MODEL}_fs
  TRAIN_PROPORTION=0.1
  TRAIN_ITERATION=300
  TEST_ITERATION=10
  TEST_INTERVAL=100
  TEST_PROPORTION=1
else
  EXP_NAME=${MODEL}
  TRAIN_PROPORTION=1
  TRAIN_ITERATION=3000
  TEST_ITERATION=10
  TEST_INTERVAL=100
  TEST_PROPORTION=1
fi
# Set output paths
MODEL_PATH=$CACHE_PATH/models/$EXP_NAME.pth
TENSORBOARD_PATH=$CACHE_PATH/tensorboard/$EXP_NAME
LOG_PATH=$CACHE_PATH/logs/$EXP_NAME.log
FIG_PATH=$CACHE_PATH/figures/$EXP_NAME
# Set model parameters
BATCH_SIZE=16
IMAGE_HEIGHT=128
IMAGE_WIDTH=384
OUTPUT_HEIGHT=16
OUTPUT_WIDTH=48
NUM_SCALE=1
NUM_CLASS=1

# Train
CUDA_VISIBLE_DEVICES=1 python $CODE_PATH/main.py --train \
  --exp_name=$EXP_NAME --model=$MODEL --batch_size=$BATCH_SIZE \
  --image_height=$IMAGE_HEIGHT --image_width=$IMAGE_WIDTH \
  --output_height=$OUTPUT_HEIGHT --output_width=$OUTPUT_WIDTH \
  --num_scale=$NUM_SCALE --num_class=$NUM_CLASS \
  --train_proportion=$TRAIN_PROPORTION --train_iteration=$TRAIN_ITERATION \
  --test_iteration=$TEST_ITERATION --test_interval=$TEST_INTERVAL \
  --save_model_path=$MODEL_PATH --tensorboard_path=$TENSORBOARD_PATH \
  |& tee $LOG_PATH

# # Test
# CUDA_VISIBLE_DEVICES=1 python $CODE_PATH/main.py --test \
#   --exp_name=$EXP_NAME --model=$MODEL --batch_size=$BATCH_SIZE \
#   --image_height=$IMAGE_HEIGHT --image_width=$IMAGE_WIDTH \
#   --output_height=$OUTPUT_HEIGHT --output_width=$OUTPUT_WIDTH \
#   --num_scale=$NUM_SCALE --num_class=$NUM_CLASS \
#   --test_proportion=$TEST_PROPORTION \
#   --init_model_path=$MODEL_PATH \
#   |& tee -a $LOG_PATH
# 
# # Predict one image and visualize
# IMAGE_NAME=$KITTI_DATA_PATH/training/image_2/000003.png
# DEPTH_NAME=$KITTI_DATA_PATH/training/disp_unsup/000003.png
# FLOW_NAME=$KITTI_DATA_PATH/training/flow_unsup/000003.png
# BOX_NAME=$KITTI_DATA_PATH/training/label_2/000003.txt
# CUDA_VISIBLE_DEVICES=1 python $CODE_PATH/main.py --visualize \
#   --exp_name=$EXP_NAME --model=$MODEL \
#   --image_height=$IMAGE_HEIGHT --image_width=$IMAGE_WIDTH \
#   --output_height=$OUTPUT_HEIGHT --output_width=$OUTPUT_WIDTH \
#   --num_scale=$NUM_SCALE --num_class=$NUM_CLASS \
#   --init_model_path=$MODEL_PATH \
#   --image_name=$IMAGE_NAME --depth_name=$DEPTH_NAME \
#   --flow_name=$FLOW_NAME --box_name=$BOX_NAME \
#   --figure_path=$FIG_PATH
# 
# # Predict a set of images and visualize
# CUDA_VISIBLE_DEVICES=1 python $CODE_PATH/main.py --visualize_all \
#   --exp_name=$EXP_NAME --model=$MODEL \
#   --image_height=$IMAGE_HEIGHT --image_width=$IMAGE_WIDTH \
#   --output_height=$OUTPUT_HEIGHT --output_width=$OUTPUT_WIDTH \
#   --num_scale=$NUM_SCALE --num_class=$NUM_CLASS \
#   --init_model_path=$MODEL_PATH \
#   --image_list=images.txt \
#   --figure_path=$FIG_PATH
