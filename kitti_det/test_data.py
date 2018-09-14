from learning_args import parse_args
from kitti_data import KittiData

import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


class DataTest(object):
    def __init__(self, data):
        self.data = data

    def test_statistics(self):
        self.data.show_basic_statistics('train')
        self.data.show_basic_statistics('test')
        self.data.show_full_statistics('train')
        self.data.show_full_statistics('test')

    def test(self):
        im, orig_im, dp, orig_dp, fl, orig_fl, box, lb, of, _, _ = self.data.get_next_batch('train')
        self.data.visualize(im, orig_im, dp, orig_dp, fl, orig_fl, box, lb, of)
        im, orig_im, dp, orig_dp, fl, orig_fl, box, lb, of, _, _ = self.data.get_next_batch('test')
        self.data.visualize(im, orig_im, dp, orig_dp, fl, orig_fl, box, lb, of)

    def test_one_image(self, image_name, depth_name, flow_name, box_name):
        im, orig_im, dp, orig_dp, fl, orig_fl, box, lb, of = \
                self.data.get_one_sample(image_name, depth_name, flow_name, box_name)
        self.data.visualize(im, orig_im, dp, orig_dp, fl, orig_fl, box, lb, of)


def main():
    args = parse_args()
    logging.info(args)
    # args.data_path = '/media/yi/DATA/data-orig/kitti/training'
    # args.image_name = '/media/yi/DATA/data-orig/kitti/training/image_2/007480.png'
    # args.depth_name = '/media/yi/DATA/data-orig/kitti/training/disp_unsup/007480.png'
    # args.flow_name = '/media/yi/DATA/data-orig/kitti/training/flow_unsup/007480.png'
    # args.box_name = '/media/yi/DATA/data-orig/kitti/training/label_2/007480.txt'

    if args.data == 'kitti':
        data = KittiData(args.data_path, args.batch_size, args.image_heights, args.image_widths,
                         args.output_heights, args.output_widths, args.num_scale,
                         args.train_proportion, args.test_proportion, args.show_statistics)
    else:
        print('Not Implemented Yet')
        return

    data_test = DataTest(data)
    data_test.test()
    data_test.test_one_image(args.image_name, args.depth_name, args.flow_name, args.box_name)
    # data_test.test_statistics()


if __name__ == '__main__':
    main()
