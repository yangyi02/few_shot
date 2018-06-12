CUR_DIR=$(pwd)

MODEL=soft_comb
EXP_NAME=soft_comb4
MODEL_PATH=$CUR_DIR/models/$EXP_NAME.pth
TENSORBOARD_PATH=$CUR_DIR/tensorboard/$EXP_NAME
LOG_PATH=$CUR_DIR/logs/$EXP_NAME.log
FIG_PATH=$CUR_DIR/figures/$EXP_NAME
BATCH_SIZE=32
IMAGE_SIZE=256
ATTENTION_SIZE=8
NUM_SCALE=2
NUM_CLASS=6
DISCRETE_DIRECT=4
TRAIN_ITERATION=2
TEST_ITERATION=1
TEST_INTERVAL=2
TEST_PROPORTION=0.003

cd ..

## Train
CUDA_VISIBLE_DEVICES=1 python main.py --train --exp_name=$EXP_NAME --data=mlt --model=$MODEL --batch_size=$BATCH_SIZE --image_size=$IMAGE_SIZE --attention_size=$ATTENTION_SIZE --num_scale=$NUM_SCALE --num_class=$NUM_CLASS --direction_type=$DISCRETE_DIRECT --show_statistics=basic --save_model_path=$MODEL_PATH --tensorboard_path=$TENSORBOARD_PATH --train_iteration=$TRAIN_ITERATION --test_iteration=$TEST_ITERATION --test_interval=$TEST_INTERVAL |& tee $LOG_PATH

## Test
CUDA_VISIBLE_DEVICES=1 python main.py --test --exp_name=$EXP_NAME --data=mlt --model=$MODEL --batch_size=$BATCH_SIZE --image_size=$IMAGE_SIZE --attention_size=$ATTENTION_SIZE --num_scale=$NUM_SCALE --num_class=$NUM_CLASS --direction_type=$DISCRETE_DIRECT --show_statistics=basic --init_model_path=$MODEL_PATH --test_proportion=$TEST_PROPORTION |& tee -a $LOG_PATH

## Visualize one image
IMAGE_NAME='/media/yi/DATA/data-orig/MLT/image/7e42db1a0bede39acb87cc8e05a90a92/000002_color.jpg'
BOX_NAME='/home/yi/code/few_shot/mlt/box/7e42db1a0bede39acb87cc8e05a90a92/000002.txt'
DEPTH_NAME='/media/yi/DATA/data-orig/MLT/depth/7e42db1a0bede39acb87cc8e05a90a92/000002_depth.png'
CUDA_VISIBLE_DEVICES=1 python main.py --visualize --exp_name=$EXP_NAME --data=mlt --model=$MODEL --image_size=$IMAGE_SIZE --attention_size=$ATTENTION_SIZE --num_scale=$NUM_SCALE --num_class=$NUM_CLASS --direction_type=$DISCRETE_DIRECT --init_model_path=$MODEL_PATH --image_name=$IMAGE_NAME --box_name=$BOX_NAME --depth_name=$DEPTH_NAME --figure_path=$FIG_PATH

## Visualize a set of images
CUDA_VISIBLE_DEVICES=1 python main.py --visualize_all --exp_name=$EXP_NAME --data=mlt --model=$MODEL --image_size=$IMAGE_SIZE --attention_size=$ATTENTION_SIZE --num_scale=$NUM_SCALE --num_class=$NUM_CLASS --direction_type=$DISCRETE_DIRECT --init_model_path=$MODEL_PATH --image_list=images.txt --figure_path=$FIG_PATH

cd $CUR_DIR
