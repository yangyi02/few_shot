from learning_args import parse_args
from kitti_data import KittiData

import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


class DataTest(object):
    def __init__(self, data):
        self.data = data

    def test(self):
        # self.data.show_basic_statistics('train')
        # self.data.show_basic_statistics('test')
        # self.data.show_full_statistics('train')
        # self.data.show_full_statistics('test')

        im, orig_im, dp, orig_dp, fl, orig_fl, box, lb, of, _, _ = self.data.get_next_batch('train')
        self.data.visualize(im, orig_im, dp, orig_dp, fl, orig_fl, box, lb, of)
        im, orig_im, dp, orig_dp, fl, orig_fl, box, lb, of, _, _ = self.data.get_next_batch('test')
        self.data.visualize(im, orig_im, dp, orig_dp, fl, orig_fl, box, lb, of)

        im, orig_im, dp, orig_dp, fl, orig_fl, box, lb, of = \
                self.data.get_one_sample(image_name, depth_name, flow_name, box_name)
        self.data.visualize(im, orig_im, dp, orig_dp, fl, orig_fl, box, lb, of)


def main():
    args = parse_args()
    logging.info(args)

    if args.data == 'kitti':
        data = KittiData(args.data_path, args.batch_size, args.image_heights, args.image_widths,
                         args.output_heights, args.output_widths, args.num_scale,
                         args.train_proportion, args.test_proportion, args.show_statistics)
    else:
        print('Not Implemented Yet')
        return

    data_test = DataTest(data)
    data_test.test()


if __name__ == '__main__':
    main()
