from learning_args import parse_args
from mlt_data import MLTData

import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


class DataTest(object):
    def __init__(self, data):
        self.data = data

    def test(self):
        self.data.show_basic_statistics('train')
        self.data.show_basic_statistics('test')
        self.data.show_full_statistics('train')
        self.data.show_full_statistics('test')

        im, orig_im, dp, orig_dp, box, direction, label, _, _ = self.data.get_next_batch('train')
        self.data.visualize(im, orig_im, dp, orig_dp, box, direction, label, 0)
        im, orig_im, dp, orig_dp, box, direction, label, _, _ = self.data.get_next_batch('test')
        self.data.visualize(im, orig_im, dp, orig_dp, box, direction, label, 0)

        image_name='/media/yi/DATA/data-orig/MLT/image/7e42db1a0bede39acb87cc8e05a90a92/000002_color.jpg'
        depth_name='/media/yi/DATA/data-orig/MLT/depth/7e42db1a0bede39acb87cc8e05a90a92/000002_depth.png'
        box_name='/home/yi/code/few_shot/mlt/box/7e42db1a0bede39acb87cc8e05a90a92/000002.txt'
        im, orig_im, dp, orig_dp, box, direction, label = \
                self.data.get_one_sample(image_name, depth_name, box_name)
        self.data.visualize(im, orig_im, dp, orig_dp, box, direction, label)


def main():
    args = parse_args()
    logging.info(args)
    if args.data == 'mlt':
        data = MLTData(args.batch_size, args.image_size, args.direction_type, args.train_proportion,
                       args.test_proportion, args.show_statistics)
    elif args.data == 'viper':
        print('Not Implemented Yet')
        return

    data_test = DataTest(data)
    data_test.test()

if __name__ == '__main__':
    main()
