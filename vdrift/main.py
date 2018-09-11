from learning_args import parse_args
from interface import SegmentInterface
from vdrift_data import VDriftData

from networks.base import BaseNet
from networks.base_3d import Base3DNet
from networks.base_2stream import Base2StreamNet

import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


def main():
    logging.info('----------------------------------------------------------------')
    logging.info('****************************************************************')
    args = parse_args()
    logging.info(args)

    if args.data == 'vdrift':
        data = VDriftData(args.data_path, args.batch_size, args.image_heights, args.image_widths,
                          args.output_height, args.output_width, args.num_scale,
                          args.train_proportion, args.test_proportion, args.show_statistics)
    else:
        print('Data not implemented yet')
        return

    if args.model == 'base':
        model = BaseNet(args.image_channel, args.num_class)
    elif args.model == 'base_3d':
        model = Base3DNet(args.image_channel, args.depth_channel, args.num_class)
    elif args.model == 'base_2stream':
        model = Base2StreamNet(args.image_channel, args.depth_channel, args.num_class)
    else:
        print('Model not implemented yet')
        return

    interface = SegmentInterface(data, model, args.learning_rate, args.train_iteration,
                                 args.test_iteration, args.test_interval, args.save_interval,
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
