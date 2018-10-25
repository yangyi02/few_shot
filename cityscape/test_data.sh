#!/bin/bash
CUR_DIR=$(dirname "$0")
source $CUR_DIR/set_env.sh

IMAGE_NAME=$DATA_PATH/cam/01_00000.png
DEPTH_NAME=$DATA_PATH/depth/01_00000.png
FLOW_X_NAME=$DATA_PATH/flow_x/01_00000.png
FLOW_Y_NAME=$DATA_PATH/flow_y/01_00000.png
SEG_NAME=$DATA_PATH/segcls/01_00000.png

python $CODE_PATH/test_data.py --data_path=$DATA_PATH --image_name=$IMAGE_NAME --depth_name=$DEPTH_NAME --flow_x_name=$FLOW_X_NAME --flow_y_name=$FLOW_Y_NAME --seg_name=$SEG_NAME
