from learning_args import parse_args
from kitti_data import KittiData, KittiDataset

import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


class DataTest(object):
    def __init__(self, data, args):
        self.data = data
        self.im_heights = args.image_heights
        self.im_widths = args.image_widths
        self.num_scale = args.num_scale
        self.ou_heights = args.output_heights
        self.ou_widths = args.output_widths

    def test_statistics(self):
        self.data.show_basic_statistics('train')
        self.data.show_basic_statistics('test')
        self.data.show_full_statistics('train')
        self.data.show_full_statistics('test')

    def test(self):
        im_h, im_w, n_s = self.im_heights, self.im_widths, self.num_scale
        ou_h, ou_w = self.ou_heights, self.ou_widths
        train_dataset = KittiDataset(self.data.train_meta, im_h, im_w, n_s, ou_h, ou_w,
                                     data_augment=True)
        for i in range(2):
            sample = train_dataset[i]
            for n in range(n_s):
                print(sample['images'][n].shape, sample['depths'][n].shape, sample['flows'][n].shape,
                      sample['heatmaps'][n].shape, sample['offsets'][n].shape)

        test_dataset = KittiDataset(self.data.test_meta, im_h, im_w, n_s, ou_h, ou_w)
        for i in range(2):
            sample = test_dataset[i]
            for n in range(n_s):
                print(sample['images'][n].shape, sample['depths'][n].shape, sample['flows'][n].shape,
                      sample['heatmaps'][n].shape, sample['offsets'][n].shape)

        train_sample = train_dataset[3]
        train_dataset.visualize(train_sample)
        test_sample = test_dataset[3]
        test_dataset.visualize(test_sample)

    def test_one_image(self, image_name, depth_name, flow_name, box_name):
        im_h, im_w, n_s = self.im_heights, self.im_widths, self.num_scale
        ou_h, ou_w = self.ou_heights, self.ou_widths
        meta = {'image': [image_name], 'depth': [depth_name], 'flow': [flow_name], 'box': [box_name]}
        meta = self.data.rearrange_annotation(meta)
        dataset = KittiDataset(meta, im_h, im_w, n_s, ou_h, ou_w)
        sample = dataset[0]
        print(sample['images'][0].shape, sample['depths'][0].shape, sample['flows'][0].shape,
              sample['heatmaps'][0].shape, sample['offsets'][0].shape)
        self.data.visualize(sample)


def main():
    args = parse_args()
    logging.info(args)
    # args.data_path = '/mnt/project/yangyi05/kitti/training'
    args.data_path = '/media/yi/DATA/data-orig/kitti/training'

    if args.data == 'kitti':
        data = KittiData(args.data_path, args.train_proportion, args.test_proportion, args.show_statistics)
    else:
        print('Not Implemented Yet')
        return

    data_test = DataTest(data, args)
    data_test.test()
    data_test.test_one_image(args.image_name, args.depth_name, args.flow_name, args.box_name)
    # data_test.test_statistics()


if __name__ == '__main__':
    main()
