from learning_args import parse_args
from cityscape_data import CityScapeData
from visualize import visualize_input, visualize_

import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


class DataTest(object):
    def __init__(self, data):
        self.data = data

    def test_statistics(self):
        self.data.show_basic_statistics()
        self.data.show_full_statistics()

    def test(self):
        sample = self.data.get_next_batch(0)
        visualize_input(sample, display=True)
        visualize_heatmap(sample, display=True)
        visualize_box(sample, display=True)


def main():
    args = parse_args()
    logging.info(args)

    if args.data == 'cityscape':
        train_data = CityScapeData(args.train_file_list, args.batch_size, args.image_heights, args.image_widths,
                             args.output_height, args.output_width, args.num_scale,
                             args.train_proportion, data_augment=False, shuffle=False)
        test_data = CityScapeData(args.train_file_list, args.batch_size, args.image_heights, args.image_widths,
                             args.output_height, args.output_width, args.num_scale,
                             args.train_proportion, data_augment=False, shuffle=False)
    else:
        print('Not Implemented Yet')
        return

    data_test = DataTest(train_data)
    data_test.test()
    data_test = DataTest(test_data)
    data_test.test()


if __name__ == '__main__':
    main()
