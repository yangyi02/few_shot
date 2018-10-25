from learning_args import parse_args
from interface import SegmentInterface
from cityscape_data import CityScapeData
from unsup_data import UnsupData
from left_right_data import LeftRightData

from networks.base import BaseNet
from networks.base_3d import Base3DNet
from networks.base_3df import Base3DFNet
from networks.base_2stream import Base2StreamNet

import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


def main():
    logging.info('----------------------------------------------------------------')
    logging.info('****************************************************************')
    args = parse_args()
    logging.info(args)

    if args.data == 'cityscape':
        train_data = CityScapeData(args.train_file_list, args.batch_size, args.image_heights,
                                   args.image_widths, args.output_height, args.output_width,
                                   args.num_scale, args.train_proportion, data_augment=False,
                                   shuffle=False)
        test_data = CityScapeData(args.test_file_list, args.batch_size, args.image_heights,
                                  args.image_widths, args.output_height, args.output_width,
                                  args.num_scale, args.test_proportion, data_augment=False,
                                  shuffle=False)
    elif args.data == 'unsup':
        train_data = UnsupData(args.train_file_list, args.batch_size, args.image_heights,
                                   args.image_widths, args.output_height, args.output_width,
                                   args.num_scale, args.train_proportion, data_augment=False,
                                   shuffle=False)
        test_data = UnsupData(args.test_file_list, args.batch_size, args.image_heights,
                                  args.image_widths, args.output_height, args.output_width,
                                  args.num_scale, args.test_proportion, data_augment=False,
                                  shuffle=False)
    elif args.data == 'left_right':
        train_data = LeftRightData(args.train_file_list, args.batch_size, args.image_heights,
                                   args.image_widths, args.output_height, args.output_width,
                                   args.num_scale, args.train_proportion, data_augment=False,
                                   shuffle=False)
        test_data = LeftRightData(args.test_file_list, args.batch_size, args.image_heights,
                                  args.image_widths, args.output_height, args.output_width,
                                  args.num_scale, args.test_proportion, data_augment=False,
                                  shuffle=False)
    else:
        print('Data not implemented yet')
        return

    if args.model == 'base':
        model = BaseNet(args.image_channel, args.num_class)
    elif args.model == 'base_3d':
        model = Base3DNet(args.image_channel, args.depth_channel, args.num_class)
    elif args.model == 'base_3df':
        model = Base3DFNet(args.image_channel, args.depth_channel, args.flow_channel, args.num_class)
    elif args.model == 'base_2stream':
        model = Base2StreamNet(args.image_channel, args.depth_channel, args.num_class)
    else:
        print('Model not implemented yet')
        return

    interface = SegmentInterface(train_data, test_data, model, args.learning_rate, args.train_epoch,
                                 args.test_interval, args.test_iteration, args.save_interval,
                                 args.init_model_path, args.save_model_path,
                                 args.tensorboard_path)

    if args.train:
        logging.info('Experiment: %s, training', args.exp_name)
        interface.train()
    elif args.test:
        logging.info('Experiment: %s, testing all', args.exp_name)
        interface.test_all()
    elif args.visualize:
        logging.info('Experiment: %s, visualizing', args.exp_name)
        interface.visualize(args.image_name, args.depth_name, args.flow_x_name, args.flow_y_name,
                            args.seg_name, args.figure_path)
    elif args.visualize_all:
        logging.info('Experiment: %s, visualizing all', args.exp_name)
        interface.visualize_all(args.image_list, args.figure_path)
    else:
        print('Unknown command')
        return


if __name__ == '__main__':
    main()
