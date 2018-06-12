CUR_DIR=$(pwd)

MODEL=base_3d
EXP_NAME=${MODEL}
MODEL_PATH=$CUR_DIR/models/$EXP_NAME.pth
TENSORBOARD_PATH=$CUR_DIR/tensorboard/$EXP_NAME
LOG_PATH=$CUR_DIR/logs/$EXP_NAME.log
FIG_PATH=$CUR_DIR/figures/$EXP_NAME
BATCH_SIZE=32
IMAGE_SIZE=256
NUM_SCALE=1
NUM_CLASS=6
TRAIN_ITERATION=2
TEST_ITERATION=1
TEST_INTERVAL=2
TEST_PROPORTION=0.003

cd ..

## Train
CUDA_VISIBLE_DEVICES=1 python main.py --train --exp_name=$EXP_NAME --data=mlt --model=$MODEL --batch_size=$BATCH_SIZE --image_size=$IMAGE_SIZE --num_scale=$NUM_SCALE --num_class=$NUM_CLASS --show_statistics=basic --save_model_path=$MODEL_PATH --tensorboard_path=$TENSORBOARD_PATH --train_iteration=$TRAIN_ITERATION --test_iteration=$TEST_ITERATION --test_interval=$TEST_INTERVAL |& tee $LOG_PATH

## Test
CUDA_VISIBLE_DEVICES=1 python main.py --test --exp_name=$EXP_NAME --data=mlt --model=$MODEL --batch_size=$BATCH_SIZE --image_size=$IMAGE_SIZE --num_scale=$NUM_SCALE --num_class=$NUM_CLASS --show_statistics=basic --init_model_path=$MODEL_PATH --test_proportion=$TEST_PROPORTION |& tee -a $LOG_PATH

cd $CUR_DIR
