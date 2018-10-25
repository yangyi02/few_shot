import argparse
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


def parse_multi_scale_parameters(args):
    image_heights, image_widths = [], []
    for i in range(args.num_scale):
        image_heights.append(int(args.image_height / pow(2, i)))
        image_widths.append(int(args.image_width / pow(2, i)))
    args.image_heights = image_heights
    args.image_widths = image_widths
    return args


def parse_args():
    arg_parser = argparse.ArgumentParser(description='depth flow for segmentation', add_help=False)
    arg_parser.add_argument('--exp_name', default='')

    arg_parser.add_argument('--train', action='store_true')
    arg_parser.add_argument('--test', action='store_true')
    arg_parser.add_argument('--visualize', action='store_true')
    arg_parser.add_argument('--visualize_all', action='store_true')

    arg_parser.add_argument('--data', default='cityscape')
    arg_parser.add_argument('--model', default='base')

    arg_parser.add_argument('--learning_rate', type=float, default=0.001)
    arg_parser.add_argument('--train_epoch', type=int, default=4)
    arg_parser.add_argument('--test_interval', type=int, default=100)
    arg_parser.add_argument('--test_iteration', type=int, default=10)
    arg_parser.add_argument('--save_interval', type=int, default=3000)

    arg_parser.add_argument('--batch_size', type=int, default=8)
    arg_parser.add_argument('--image_height', type=int, default=256)
    arg_parser.add_argument('--image_width', type=int, default=512)
    arg_parser.add_argument('--output_height', type=int, default=256)
    arg_parser.add_argument('--output_width', type=int, default=512)
    arg_parser.add_argument('--image_channel', type=int, default=3)
    arg_parser.add_argument('--depth_channel', type=int, default=1)
    arg_parser.add_argument('--flow_channel', type=int, default=2)
    arg_parser.add_argument('--num_scale', type=int, default=1)
    arg_parser.add_argument('--num_class', type=int, default=1)
    arg_parser.add_argument('--train_proportion', type=float, default=1.0)
    arg_parser.add_argument('--test_proportion', type=float, default=1.0)

    arg_parser.add_argument('--show_statistics', default='')

    arg_parser.add_argument('--init_model_path', default='')
    arg_parser.add_argument('--save_model_path', default='')
    arg_parser.add_argument('--tensorboard_path', default='tensorboard/')

    arg_parser.add_argument('--train_file_list', default='')
    arg_parser.add_argument('--test_file_list', default='')
    arg_parser.add_argument('--image_name', default='')
    arg_parser.add_argument('--depth_name', default='')
    arg_parser.add_argument('--flow_name', default='')
    arg_parser.add_argument('--seg_name', default='')
    arg_parser.add_argument('--image_list', default='')
    arg_parser.add_argument('--figure_path', default='')

    args = arg_parser.parse_args()
    args = parse_multi_scale_parameters(args)

    return args
