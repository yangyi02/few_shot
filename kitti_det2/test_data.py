from learning_args import parse_args
from kitti_data import KittiData, KittiDataLoader, read_box
from visualize import visualize_input, visualize_box, visualize_heatmap

import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


class DataTest(object):
    def __init__(self, data):
        self.data = data

    def test(self):
        sample = self.data.get_next_batch(0)
        visualize_input(sample, display=True)
        visualize_heatmap(sample, display=True)
        visualize_box(sample, display=True)


def main():
    args = parse_args()
    logging.info(args)
    # args.data_path = '/media/yi/DATA/data-orig/kitti/training'
    # args.image_name = '/media/yi/DATA/data-orig/kitti/training/image_2/007480.png'
    # args.depth_name = '/media/yi/DATA/data-orig/kitti/training/disp_unsup/007480.png'
    # args.flow_name = '/media/yi/DATA/data-orig/kitti/training/flow_unsup/007480.png'
    # args.box_name = '/media/yi/DATA/data-orig/kitti/training/label_2/007480.txt'

    if args.data == 'kitti':
        data = KittiData(args.data_path, args.train_proportion, args.test_proportion)
        train_meta = data.train_meta
        train_data = KittiDataLoader(train_meta, args.batch_size, args.image_heights,
                                     args.image_widths, args.output_heights, args.output_widths,
                                     args.num_scale, data_augment=True, shuffle=True)
        data_test = DataTest(train_data)
        data_test.test()

        test_meta = data.test_meta
        test_data = KittiDataLoader(test_meta, args.batch_size, args.image_heights,
                                    args.image_widths, args.output_heights, args.output_widths,
                                    args.num_scale)
        data_test = DataTest(test_data)
        data_test.test()

        meta = {'image': [args.image_name], 'depth': [args.depth_name], 'flow': [args.flow_name],
                'box': [read_box(args.box_name)]}
        data = KittiDataLoader(meta, args.batch_size, args.image_heights, args.image_widths,
                               args.output_heights, args.output_widths, args.num_scale)
        data_test = DataTest(data)
        data_test.test()
    else:
        print('Not Implemented Yet')
        return


if __name__ == '__main__':
    main()
