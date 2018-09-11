#!/bin/bash
CUR_DIR=$(dirname "$0")
source $CUR_DIR/../set_env.sh

MODEL=base_3d

# Mode can be test mode, fast training mode or slow training mode
# unit_test mode test the shell script, make sure training and testing works
# fast mode train the model with fewer data and fewer iterations
# full mode train the model with full data and full iterations
MODE=$1
if [ "$MODE" = "unittest" ]; then
  EXP_NAME=${MODEL}_unittest
  TRAIN_PROPORTION=0.1
  TRAIN_ITERATION=2
  TEST_INTERVAL=2
  TEST_ITERATION=1
  TEST_PROPORTION=0.05
elif [ "$MODE" = "fast" ]; then
  EXP_NAME=${MODEL}_fs
  TRAIN_PROPORTION=0.1
  TRAIN_ITERATION=300
  TEST_INTERVAL=50
  TEST_ITERATION=10
  TEST_PROPORTION=1
else
  EXP_NAME=${MODEL}
  TRAIN_PROPORTION=1
  TRAIN_ITERATION=3000
  TEST_INTERVAL=100
  TEST_ITERATION=10
  TEST_PROPORTION=1
fi
# Set output paths
MODEL_PATH=$CACHE_PATH/models/$EXP_NAME.pth
TENSORBOARD_PATH=$CACHE_PATH/tensorboard/$EXP_NAME
LOG_PATH=$CACHE_PATH/logs/$EXP_NAME.log
FIG_PATH=$CACHE_PATH/figures/$EXP_NAME
# Set model parameters
BATCH_SIZE=8
IMAGE_HEIGHT=240
IMAGE_WIDTH=320
OUTPUT_HEIGHT=240
OUTPUT_WIDTH=320
NUM_SCALE=1
NUM_CLASS=8

# Train
CUDA_VISIBLE_DEVICES=1 python $CODE_PATH/main.py --train \
  --exp_name=$EXP_NAME --model=$MODEL --data_path=$DATA_PATH \
  --batch_size=$BATCH_SIZE \
  --image_height=$IMAGE_HEIGHT --image_width=$IMAGE_WIDTH \
  --output_height=$OUTPUT_HEIGHT --output_width=$OUTPUT_WIDTH \
  --num_scale=$NUM_SCALE --num_class=$NUM_CLASS \
  --train_proportion=$TRAIN_PROPORTION --train_iteration=$TRAIN_ITERATION \
  --test_interval=$TEST_INTERVAL --test_iteration=$TEST_ITERATION \
  --save_model_path=$MODEL_PATH --tensorboard_path=$TENSORBOARD_PATH \
  |& tee $LOG_PATH

# Test
CUDA_VISIBLE_DEVICES=1 python $CODE_PATH/main.py --test \
  --exp_name=$EXP_NAME --model=$MODEL --data_path=$DATA_PATH \
  --batch_size=$BATCH_SIZE \
  --image_height=$IMAGE_HEIGHT --image_width=$IMAGE_WIDTH \
  --output_height=$OUTPUT_HEIGHT --output_width=$OUTPUT_WIDTH \
  --num_scale=$NUM_SCALE --num_class=$NUM_CLASS \
  --test_proportion=$TEST_PROPORTION \
  --init_model_path=$MODEL_PATH \
  |& tee -a $LOG_PATH

# Predict one image and visualize
IMAGE_NAME=$DATA_PATH/cam/01_00000.png
DEPTH_NAME=$DATA_PATH/depth/01_00000.png
FLOW_X_NAME=$DATA_PATH/flow_x/01_00000.png
FLOW_Y_NAME=$DATA_PATH/flow_y/01_00000.png
SEG_NAME=$DATA_PATH/segcls/01_00000.png
CUDA_VISIBLE_DEVICES=1 python $CODE_PATH/main.py --visualize \
  --exp_name=$EXP_NAME --model=$MODEL --data_path=$DATA_PATH \
  --image_height=$IMAGE_HEIGHT --image_width=$IMAGE_WIDTH \
  --output_height=$OUTPUT_HEIGHT --output_width=$OUTPUT_WIDTH \
  --num_scale=$NUM_SCALE --num_class=$NUM_CLASS \
  --init_model_path=$MODEL_PATH \
  --image_name=$IMAGE_NAME --depth_name=$DEPTH_NAME \
  --flow_x_name=$FLOW_X_NAME --flow_y_name=$FLOW_Y_NAME --seg_name=$SEG_NAME \
  --figure_path=$FIG_PATH

# Predict a set of images and visualize
CUDA_VISIBLE_DEVICES=1 python $CODE_PATH/main.py --visualize_all \
  --exp_name=$EXP_NAME --model=$MODEL --data_path=$DATA_PATH \
  --image_height=$IMAGE_HEIGHT --image_width=$IMAGE_WIDTH \
  --output_height=$OUTPUT_HEIGHT --output_width=$OUTPUT_WIDTH \
  --num_scale=$NUM_SCALE --num_class=$NUM_CLASS \
  --init_model_path=$MODEL_PATH \
  --image_list=$CODE_PATH/images.txt \
  --figure_path=$FIG_PATH
