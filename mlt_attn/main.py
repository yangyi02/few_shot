from learning_args import parse_args
from interface_soft import SoftAttnInterface
from interface_hard import HardAttnInterface
from mlt_data import MLTData

from networks.base import BaseNet
from networks.base_direct import BaseDirectNet
from networks.base_3d import Base3DNet
from networks.base_direct_3d import BaseDirect3DNet
from networks.base_2stream import Base2StreamNet
from networks.base_direct_2stream import BaseDirect2StreamNet
from networks.hard_gt_attn import HardGtAttnNet
from networks.hard_direct import HardDirectNet
from networks.hard_gt_attn_3d import HardGtAttn3DNet
from networks.hard_gt_attn_2stream import HardGtAttn2StreamNet
from networks.soft_attn import SoftAttnNet
from networks.soft_direct import SoftDirectNet
from networks.soft_comb import SoftCombNet
from networks.soft_attn_3d import SoftAttn3DNet
from networks.soft_direct_3d import SoftDirect3DNet
from networks.soft_comb_3d import SoftComb3DNet
from networks.soft_attn_2stream import SoftAttn2StreamNet
from networks.soft_direct_2stream import SoftDirect2StreamNet
from networks.soft_comb_2stream import SoftComb2StreamNet

import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


def main():
    logging.info('----------------------------------------------------------------')
    logging.info('****************************************************************')
    args = parse_args()
    logging.info(args)

    if args.data == 'mlt':
        data = MLTData(args.batch_size, args.image_size, args.direction_type, args.train_proportion,
                       args.test_proportion, args.show_statistics)
    elif args.data == 'viper':
        print('Not Implemented Yet')
        return

    if args.model == 'base':
        model = BaseNet(args.image_channel, args.num_class)
    elif args.model == 'base_direct':
        model = BaseDirectNet(args.image_channel, args.direction_dim, args.num_class)
    elif args.model == 'base_3d':
        model = Base3DNet(args.image_channel, args.depth_channel, args.num_class)
    elif args.model == 'base_direct_3d':
        model = BaseDirect3DNet(args.image_channel, args.depth_channel, args.direction_dim, args.num_class)
    elif args.model == 'base_2stream':
        model = Base2StreamNet(args.image_channel, args.depth_channel, args.num_class)
    elif args.model == 'base_direct_2stream':
        model = BaseDirect2StreamNet(args.image_channel, args.depth_channel, args.direction_dim, args.num_class)
    elif args.model == 'hard_gt_attn':
        model = HardGtAttnNet(args.image_size[0], args.image_channel, args.num_class)
    elif args.model == 'hard_direct':
        model = HardDirectNet(args.image_size[0], args.image_channel, args.direction_dim, args.num_class)
    elif args.model == 'hard_gt_attn_3d':
        model = HardGtAttn3DNet(args.image_size[0], args.image_channel, args.depth_channel, args.num_class)
    elif args.model == 'hard_gt_attn_2stream':
        model = HardGtAttn2StreamNet(args.image_size[0], args.image_channel, args.depth_channel, args.num_class)
    elif args.model == 'soft_attn':
        model = SoftAttnNet(args.attention_size, args.image_channel, args.num_class)
    elif args.model == 'soft_direct':
        model = SoftDirectNet(args.attention_size, args.image_channel, args.direction_dim, args.num_class)
    elif args.model == 'soft_comb':
        model = SoftCombNet(args.attention_size, args.image_channel, args.direction_dim, args.num_class)
    elif args.model == 'soft_attn_3d':
        model = SoftAttn3DNet(args.attention_size, args.image_channel, args.depth_channel, args.num_class)
    elif args.model == 'soft_direct_3d':
        model = SoftDirect3DNet(args.attention_size, args.image_channel, args.depth_channel, args.direction_dim, args.num_class)
    elif args.model == 'soft_comb_3d':
        model = SoftComb3DNet(args.attention_size, args.image_channel, args.depth_channel, args.direction_dim, args.num_class)
    elif args.model == 'soft_attn_2stream':
        model = SoftAttn2StreamNet(args.attention_size, args.image_channel, args.depth_channel, args.num_class)
    elif args.model == 'soft_direct_2stream':
        model = SoftDirect2StreamNet(args.attention_size, args.image_channel, args.depth_channel, args.direction_dim, args.num_class)
    elif args.model == 'soft_comb_2stream':
        model = SoftComb2StreamNet(args.attention_size, args.image_channel, args.depth_channel, args.direction_dim, args.num_class)

    if args.attention_type == 'soft':
        interface = SoftAttnInterface(data, model, args.learning_rate, args.train_iteration, args.test_iteration,
                                      args.test_interval, args.save_interval, args.init_model_path,
                                      args.save_model_path, args.tensorboard_path)
    elif args.attention_type == 'hard':
        interface = HardAttnInterface(data, model, args.learning_rate, args.train_iteration, args.test_iteration,
                                      args.test_interval, args.save_interval, args.init_model_path,
                                      args.save_model_path, args.tensorboard_path)

    if args.train:
        logging.info('Experiment: %s, training', args.exp_name)
        interface.train()
    elif args.test:
        logging.info('Experiment: %s, testing all', args.exp_name)
        interface.test_all()
    elif args.visualize:
        logging.info('Experiment: %s, visualizing', args.exp_name)
        interface.visualize(args.image_name, args.depth_name, args.box_name, args.figure_path)
    elif args.visualize_all:
        logging.info('Experiment: %s, visualizing all', args.exp_name)
        interface.visualize_all(args.image_list, args.figure_path)

if __name__ == '__main__':
    main()
