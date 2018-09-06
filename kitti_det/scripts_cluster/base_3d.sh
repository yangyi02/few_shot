#!/bin/bash
CUR_DIR=$(dirname "$0")
source $CUR_DIR/../set_env.sh

MODEL=base_3d

# Mode can be test mode, fast training mode or slow training mode
# unit_test mode test the shell script, make sure training and testing works
# fast mode train the model with fewer data and fewer iterations
# full mode train the model with full data and full iterations
MODE=$1
if [ "$MODE" = "unit_test" ]; then
  EXP_NAME=${MODEL}_test
  TRAIN_PROPORTION=0.1
  TRAIN_ITERATION=2
  TEST_INTERVAL=2
  TEST_ITERATION=1
  TEST_PROPORTION=0.05
elif [ "$MODE" = "fast" ]; then
  EXP_NAME=${MODEL}_fs
  TRAIN_PROPORTION=0.1
  TRAIN_ITERATION=300
  TEST_INTERVAL=100
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
ERROR_PATH=$CACHE_PATH/logs/${EXP_NAME}_error.log
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
COMMAND="python $CODE_PATH/main.py --train \
  --exp_name=$EXP_NAME --model=$MODEL --data_path=$DATA_PATH \
  --batch_size=$BATCH_SIZE \
  --image_height=$IMAGE_HEIGHT --image_width=$IMAGE_WIDTH \
  --output_height=$OUTPUT_HEIGHT --output_width=$OUTPUT_WIDTH \
  --num_scale=$NUM_SCALE --num_class=$NUM_CLASS \
  --train_proportion=$TRAIN_PROPORTION --train_iteration=$TRAIN_ITERATION \
  --test_interval=$TEST_INTERVAL --test_iteration=$TEST_ITERATION \
  --save_model_path=$MODEL_PATH --tensorboard_path=$TENSORBOARD_PATH"
echo $COMMAND
sbatch --partition=1080Ti_dbg --gres=gpu:1 --job-name=$EXP_NAME --cpus-per-task=4 --ntasks=1 \
  --output=$LOG_PATH --error=$ERROR_PATH --wrap="$COMMAND"

# Test
COMMAND="python $CODE_PATH/main.py --test \
  --exp_name=$EXP_NAME --model=$MODEL --data_path=$DATA_PATH \
  --batch_size=$BATCH_SIZE \
  --image_height=$IMAGE_HEIGHT --image_width=$IMAGE_WIDTH \
  --output_height=$OUTPUT_HEIGHT --output_width=$OUTPUT_WIDTH \
  --num_scale=$NUM_SCALE --num_class=$NUM_CLASS \
  --test_proportion=$TEST_PROPORTION \
  --init_model_path=$MODEL_PATH"
echo $COMMAND
sbatch --partition=1080Ti_dbg --gres=gpu:1 --job-name=$EXP_NAME --cpus-per-task=4 --ntasks=1 \
  --output=$LOG_PATH --error=$ERROR_PATH --wrap="$COMMAND"

# Predict one image and visualize
IMAGE_NAME=$DATA_PATH/image_2/000003.png
DEPTH_NAME=$DATA_PATH/disp_unsup/000003.png
FLOW_NAME=$DATA_PATH/flow_unsup/000003.png
BOX_NAME=$DATA_PATH/label_2/000003.txt
COMMAND="python $CODE_PATH/main.py --visualize \
  --exp_name=$EXP_NAME --model=$MODEL --data_path=$DATA_PATH \
  --image_height=$IMAGE_HEIGHT --image_width=$IMAGE_WIDTH \
  --output_height=$OUTPUT_HEIGHT --output_width=$OUTPUT_WIDTH \
  --num_scale=$NUM_SCALE --num_class=$NUM_CLASS \
  --init_model_path=$MODEL_PATH \
  --image_name=$IMAGE_NAME --depth_name=$DEPTH_NAME \
  --flow_name=$FLOW_NAME --box_name=$BOX_NAME \
  --figure_path=$FIG_PATH"
echo $COMMAND
sbatch --partition=1080Ti_dbg --gres=gpu:1 --job-name=$EXP_NAME --cpus-per-task=4 --ntasks=1 \
  --output=$LOG_PATH --error=$ERROR_PATH --wrap="$COMMAND"

# Predict a set of images and visualize
COMMAND="python $CODE_PATH/main.py --visualize_all \
  --exp_name=$EXP_NAME --model=$MODEL --data_path=$DATA_PATH \
  --image_height=$IMAGE_HEIGHT --image_width=$IMAGE_WIDTH \
  --output_height=$OUTPUT_HEIGHT --output_width=$OUTPUT_WIDTH \
  --num_scale=$NUM_SCALE --num_class=$NUM_CLASS \
  --init_model_path=$MODEL_PATH \
  --image_list=$CODE_PATH/images.txt \
  --figure_path=$FIG_PATH"
echo $COMMAND
sbatch --partition=1080Ti_dbg --gres=gpu:1 --job-name=$EXP_NAME --cpus-per-task=4 --ntasks=1 \
  --output=$LOG_PATH --error=$ERROR_PATH --wrap="$COMMAND"
