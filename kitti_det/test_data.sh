#!/bin/bash
CUR_DIR=$(dirname "$0")
source $CUR_DIR/set_env.sh

IMAGE_NAME=$DATA_PATH/image_2/000003.png
DEPTH_NAME=$DATA_PATH/disp_unsup/000003.png
FLOW_NAME=$DATA_PATH/flow_unsup/000003.png
BOX_NAME=$DATA_PATH/label_2/000003.txt

python $CODE_PATH/test_data.py --data_path=$KITTI_DATA_PATH --image_name=$IMAGE_NAME --depth_name=$DEPTH_NAME --flow_name=$FLOW_NAME --box_name=$BOX_NAME

