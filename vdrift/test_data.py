from learning_args import parse_args
from vdrift_data import VDriftData

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
        im, orig_im, dp, orig_dp, fl, orig_fl, seg, orig_seg, _, _ = self.data.get_next_batch('train')
        self.data.visualize(im, orig_im, dp, orig_dp, fl, orig_fl, seg, orig_seg)
        im, orig_im, dp, orig_dp, fl, orig_fl, seg, orig_seg, _, _ = self.data.get_next_batch('test')
        self.data.visualize(im, orig_im, dp, orig_dp, fl, orig_fl, seg, orig_seg)

    def test_one_image(self, image_name, depth_name, flow_x_name, flow_y_name, seg_name):
        im, orig_im, dp, orig_dp, fl, orig_fl, seg, orig_seg = \
                self.data.get_one_sample(image_name, depth_name, flow_x_name, flow_y_name, seg_name)
        self.data.visualize(im, orig_im, dp, orig_dp, fl, orig_fl, seg, orig_seg)


def main():
    args = parse_args()
    logging.info(args)

    # args.data_path = '/media/yi/DATA/data-orig/vdrift'
    # args.image_name = '/media/yi/DATA/data-orig/vdrift/cam/02_00000.png'
    # args.depth_name = '/media/yi/DATA/data-orig/vdrift/depth/02_00000.png'
    # args.flow_x_name = '/media/yi/DATA/data-orig/vdrift/flow_x/02_00000.png'
    # args.flow_y_name = '/media/yi/DATA/data-orig/vdrift/flow_y/02_00000.png'
    # args.seg_name = '/media/yi/DATA/data-orig/vdrift/segcls/02_00000.png'

    if args.data == 'vdrift':
        data = VDriftData(args.data_path, args.batch_size, args.image_heights, args.image_widths,
                          args.output_height, args.output_width, args.num_scale,
                          args.train_proportion, args.test_proportion, args.show_statistics)
    else:
        print('Not Implemented Yet')
        return

    data_test = DataTest(data)
    data_test.test()
    data_test.test_one_image(args.image_name, args.depth_name, args.flow_x_name, args.flow_y_name, args.seg_name)
    # data_test.test_statistics()


if __name__ == '__main__':
    main()
