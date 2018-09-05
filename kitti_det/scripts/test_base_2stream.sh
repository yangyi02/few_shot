CUR_DIR=$(pwd)

MODEL=base_2stream
EXP_NAME=${MODEL}
MODEL_PATH=$CUR_DIR/models/$EXP_NAME.pth
TENSORBOARD_PATH=$CUR_DIR/tensorboard/$EXP_NAME
LOG_PATH=$CUR_DIR/logs/$EXP_NAME.log
FIG_PATH=$CUR_DIR/figures/$EXP_NAME
BATCH_SIZE=16
IMAGE_HEIGHT=128
IMAGE_WIDTH=384
OUTPUT_HEIGHT=16
OUTPUT_WIDTH=48
NUM_SCALE=1
NUM_CLASS=1
TRAIN_ITERATION=2
TEST_ITERATION=1
TEST_INTERVAL=2
TEST_PROPORTION=0.05

cd ..

## Train
CUDA_VISIBLE_DEVICES=1 python main.py --train --exp_name=$EXP_NAME --data=kitti --model=$MODEL --batch_size=$BATCH_SIZE --image_height=$IMAGE_HEIGHT --image_width=$IMAGE_WIDTH --output_height=$OUTPUT_HEIGHT --output_width=$OUTPUT_WIDTH --num_scale=$NUM_SCALE --num_class=$NUM_CLASS --show_statistics=basic --save_model_path=$MODEL_PATH --tensorboard_path=$TENSORBOARD_PATH --train_iteration=$TRAIN_ITERATION --test_iteration=$TEST_ITERATION --test_interval=$TEST_INTERVAL |& tee $LOG_PATH

## Test
CUDA_VISIBLE_DEVICES=1 python main.py --test --exp_name=$EXP_NAME --data=kitti --model=$MODEL --batch_size=$BATCH_SIZE --image_height=$IMAGE_HEIGHT --image_width=$IMAGE_WIDTH --output_height=$OUTPUT_HEIGHT --output_width=$OUTPUT_WIDTH --num_scale=$NUM_SCALE --num_class=$NUM_CLASS --show_statistics=basic --init_model_path=$MODEL_PATH --test_proportion=$TEST_PROPORTION |& tee -a $LOG_PATH

cd $CUR_DIR
