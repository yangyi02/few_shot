import os
import numpy as np
from PIL import Image
import cv2
import flowlib

import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


def read_box(box_name):
    class_map = {'Background': 0, 'Car': 1, 'Van': 2, 'Truck': 3, 'Pedestrian': 4,
                 'Person_sitting': 5, 'Cyclist': 6, 'Tram': 7, 'Misc': 8, 'DontCare': 9}
    box_and_label = []
    with open(box_name) as txt_file:
        box_info = txt_file.readlines()
    for row in box_info:
        row = row.strip().split(' ')
        if not row[0] in ['Car']:  # Only take car objects right now
            continue
        row[0] = class_map[row[0]]
        box_and_label.append(row)
    box_and_label = np.array(box_and_label).astype(np.float)
    if box_and_label.shape[0] == 0:
        box = np.zeros((0, 5))
    else:
        box = box_and_label[:, 4:8]
        label = box_and_label[:, 0]
        box = np.concatenate((box, label[:, None]), axis=1)
    return box


class KittiData(object):
    def __init__(self, data_path, train_proportion=1, test_proportion=1):
        self.name = 'kitti'
        self.image_dir = os.path.join(data_path, 'image_2')
        self.depth_dir = os.path.join(data_path, 'disp_unsup')
        self.flow_dir = os.path.join(data_path, 'flow_unsup')
        self.box_dir = os.path.join(data_path, 'label_2')

        self.train_proportion = train_proportion
        self.test_proportion = test_proportion
        self.train_test_split = 0.8

        self.class_map = {'Background': 0, 'Car': 1, 'Van': 2, 'Truck': 3, 'Pedestrian': 4,
                          'Person_sitting': 5, 'Cyclist': 6, 'Tram': 7, 'Misc': 8, 'DontCare': 9}
        self.inverse_class_map = {0: 'Background', 1: 'Car', 2: 'Van', 3: 'Truck', 4: 'Pedestrian',
                                  5: 'Person_sitting', 6: 'Cyclist', 7: 'Tram', 8: 'Misc',
                                  9: 'DontCare'}

        self.meta = self.get_meta()
        self.train_meta, self.test_meta = self.split_train_test()
        logging.info('number of training image: %d, number of testing image: %d',
                     len(self.train_meta['image']), len(self.test_meta['image']))

    def get_meta(self):
        box_files = os.listdir(os.path.join(self.box_dir))
        box_files = sorted(box_files)
        img_files, depth_files, flow_files = [], [], []
        for f in box_files:
            file_name, file_ext = os.path.splitext(f)
            img_files.append(file_name + '.png')
            depth_files.append(file_name + '.png')
            flow_files.append(file_name + '.png')
        meta = dict()
        meta['image'] = [os.path.join(self.image_dir, f) for f in img_files]
        meta['depth'] = [os.path.join(self.depth_dir, f) for f in depth_files]
        meta['flow'] = [os.path.join(self.flow_dir, f) for f in flow_files]
        meta['box'] = [read_box(os.path.join(self.box_dir, f)) for f in box_files]
        return meta

    def split_train_test(self):
        # Split train / test
        num_image, thresh = len(self.meta['image']), self.train_test_split
        train_meta = {'image': self.meta['image'][0:int(num_image * thresh)],
                      'depth': self.meta['depth'][0:int(num_image * thresh)],
                      'flow': self.meta['flow'][0:int(num_image * thresh)],
                      'box': self.meta['box'][0:int(num_image * thresh)]}
        test_meta = {'image': self.meta['image'][int(num_image * thresh):],
                     'depth': self.meta['depth'][int(num_image * thresh):],
                     'flow': self.meta['flow'][int(num_image * thresh):],
                     'box': self.meta['box'][int(num_image * thresh):]}
        # Take a proportion of train or test data if needed
        num_train_image = len(train_meta['image'])
        train_meta['image'] = train_meta['image'][0:int(num_train_image * self.train_proportion)]
        train_meta['depth'] = train_meta['depth'][0:int(num_train_image * self.train_proportion)]
        train_meta['flow'] = train_meta['flow'][0:int(num_train_image * self.train_proportion)]
        train_meta['box'] = train_meta['box'][0:int(num_train_image * self.train_proportion)]
        num_test_image = len(test_meta['image'])
        test_meta['image'] = test_meta['image'][0:int(num_test_image * self.test_proportion)]
        test_meta['depth'] = test_meta['depth'][0:int(num_test_image * self.test_proportion)]
        test_meta['flow'] = test_meta['flow'][0:int(num_test_image * self.test_proportion)]
        test_meta['box'] = test_meta['box'][0:int(num_test_image * self.test_proportion)]
        return train_meta, test_meta


class KittiDataLoader(object):
    def __init__(self, meta, batch_size=8, image_heights=[128], image_widths=[384],
                 output_heights=[16], output_widths=[48], num_scale=1,
                 data_augment=False, shuffle=False):
        self.meta = meta
        self.batch_size = batch_size
        self.im_heights = image_heights
        self.im_widths = image_widths
        self.num_scale = num_scale
        self.output_heights = output_heights
        self.output_widths = output_widths
        self.orig_im_size = [320, 960]
        self.data_augment = data_augment
        self.num_data = len(meta['image'])
        self.num_batch = int(np.ceil(self.num_data * 1.0 / batch_size))
        self.last_batch_size = self.num_data - (self.num_batch - 1) * batch_size
        if shuffle:
            self.sample_index = np.random.permutation(self.num_data)
        else:
            self.sample_index = np.arange(0, self.num_data)

    def shuffle(self):
        self.sample_index = np.random.permutation(self.num_data)

    def get_next_batch(self, batch_idx):
        batch_size, orig_im_size = self.batch_size, self.orig_im_size
        if batch_idx == self.num_batch - 1:
            batch_size = self.last_batch_size
        elif batch_idx >= self.num_batch:
            print('batch index larger than the total number of batches')
            return
        im_heights, im_widths, num_scale = self.im_heights, self.im_widths, self.num_scale
        ou_heights, ou_widths = self.output_heights, self.output_widths
        images = [np.zeros((batch_size, im_heights[i], im_widths[i], 3)) for i in range(num_scale)]
        depths = [np.zeros((batch_size, im_heights[i], im_widths[i], 1)) for i in range(num_scale)]
        flows = [np.zeros((batch_size, im_heights[i], im_widths[i], 3)) for i in range(num_scale)]
        heatmaps = [np.zeros((batch_size, ou_heights[i], ou_widths[i], 1)) for i in
                    range(num_scale)]
        offsets = [np.zeros((batch_size, ou_heights[i], ou_widths[i], 4)) for i in range(num_scale)]

        orig_image = np.zeros((batch_size, orig_im_size[0], orig_im_size[1], 3))
        orig_depth = np.zeros((batch_size, orig_im_size[0], orig_im_size[1], 1))
        orig_flow = np.zeros((batch_size, orig_im_size[0], orig_im_size[1], 3))
        boxes = []

        for n in range(batch_size):
            data_idx = self.sample_index[batch_idx * self.batch_size + n]
            image = np.array(Image.open(self.meta['image'][data_idx]))
            image = image / 255.0
            depth = flowlib.read_disp_png(self.meta['depth'][data_idx])
            flow = flowlib.read_flow_png(self.meta['flow'][data_idx])
            box = np.array(self.meta['box'][data_idx])

            if self.data_augment:
                image, depth, flow, box = self.flip_image(image, depth, flow, box)
                image, depth, flow, box = self.resize_crop_image(image, depth, flow, box)

            for i in range(num_scale):
                images[i][n, :, :, :] = cv2.resize(image, (im_widths[i], im_heights[i]),
                                                   interpolation=cv2.INTER_AREA)
                depths[i][n, :, :, 0] = cv2.resize(depth, (im_widths[i], im_heights[i]),
                                                   interpolation=cv2.INTER_AREA)
                flows[i][n, :, :, 0:2] = cv2.resize(flow[:, :, 0:2], (im_widths[i], im_heights[i]),
                                                    interpolation=cv2.INTER_AREA)
                flows[i][n, :, :, 2] = cv2.resize(flow[:, :, 2], (im_widths[i], im_heights[i]),
                                                  interpolation=cv2.INTER_NEAREST)
            im_height, im_width = image.shape[0], image.shape[1]
            box = self.rescale_box(box, im_height, im_width)
            boxes.append(box)
            xb, yb, wb, hb = self.get_box_center_size(box)

            for k in range(box.shape[0]):
                for i in range(num_scale):
                    if wb[k] < 2.0 / ou_widths[i] or hb[k] < 2.0 / ou_heights[i]:
                        continue

                    x = np.int(np.floor(xb[k] * ou_widths[i]))
                    y = np.int(np.floor(yb[k] * ou_heights[i]))
                    if x < 0 or x >= ou_widths[i] or y < 0 or y >= ou_heights[i]:
                        continue
                    heatmaps[i][n, y, x, 0] = 1  # Only works for car detection
                    offsets[i][n, y, x, 0] = box[k, 0] - x * 1.0 / ou_widths[i]
                    offsets[i][n, y, x, 1] = box[k, 1] - y * 1.0 / ou_heights[i]
                    offsets[i][n, y, x, 2] = box[k, 2] - x * 1.0 / ou_widths[i]
                    offsets[i][n, y, x, 3] = box[k, 3] - y * 1.0 / ou_heights[i]

                    x = np.int(np.floor(xb[k] * ou_widths[i]))
                    y = np.int(np.ceil(yb[k] * ou_heights[i]))
                    if x < 0 or x >= ou_widths[i] or y < 0 or y >= ou_heights[i]:
                        continue
                    heatmaps[i][n, y, x, 0] = 1  # Only works for car detection
                    offsets[i][n, y, x, 0] = box[k, 0] - x * 1.0 / ou_widths[i]
                    offsets[i][n, y, x, 1] = box[k, 1] - y * 1.0 / ou_heights[i]
                    offsets[i][n, y, x, 2] = box[k, 2] - x * 1.0 / ou_widths[i]
                    offsets[i][n, y, x, 3] = box[k, 3] - y * 1.0 / ou_heights[i]

                    x = np.int(np.ceil(xb[k] * ou_widths[i]))
                    y = np.int(np.floor(yb[k] * ou_heights[i]))
                    if x < 0 or x >= ou_widths[i] or y < 0 or y >= ou_heights[i]:
                        continue
                    heatmaps[i][n, y, x, 0] = 1  # Only works for car detection
                    offsets[i][n, y, x, 0] = box[k, 0] - x * 1.0 / ou_widths[i]
                    offsets[i][n, y, x, 1] = box[k, 1] - y * 1.0 / ou_heights[i]
                    offsets[i][n, y, x, 2] = box[k, 2] - x * 1.0 / ou_widths[i]
                    offsets[i][n, y, x, 3] = box[k, 3] - y * 1.0 / ou_heights[i]

                    x = np.int(np.ceil(xb[k] * ou_widths[i]))
                    y = np.int(np.ceil(yb[k] * ou_heights[i]))
                    if x < 0 or x >= ou_widths[i] or y < 0 or y >= ou_heights[i]:
                        continue
                    heatmaps[i][n, y, x, 0] = 1  # Only works for car detection
                    offsets[i][n, y, x, 0] = box[k, 0] - x * 1.0 / ou_widths[i]
                    offsets[i][n, y, x, 1] = box[k, 1] - y * 1.0 / ou_heights[i]
                    offsets[i][n, y, x, 2] = box[k, 2] - x * 1.0 / ou_widths[i]
                    offsets[i][n, y, x, 3] = box[k, 3] - y * 1.0 / ou_heights[i]

            if self.data_augment:
                orig_image[n, :, :, :] = image
                orig_depth[n, :, :, 0] = depth
                orig_flow[n, :, :, :] = flow
            else:
                orig_image[n, :, :, :] = cv2.resize(image, (orig_im_size[1], orig_im_size[0]),
                                                    interpolation=cv2.INTER_AREA)
                orig_depth[n, :, :, 0] = cv2.resize(depth, (orig_im_size[1], orig_im_size[0]),
                                                    interpolation=cv2.INTER_AREA)
                orig_flow[n, :, :, 0:2] = cv2.resize(flow[:, :, 0:2], (orig_im_size[1], orig_im_size[0]),
                                                     interpolation=cv2.INTER_AREA)
                orig_flow[n, :, :, 2] = cv2.resize(flow[:, :, 2], (orig_im_size[1], orig_im_size[0]),
                                                   interpolation=cv2.INTER_NEAREST)
        sample = {'images': images, 'flows': flows, 'depths': depths, 'heatmaps': heatmaps,
                  'offsets': offsets, 'orig_image': orig_image, 'orig_depth': orig_depth,
                  'orig_flow': orig_flow, 'boxes': boxes}
        return sample

    def flip_image(self, image, depth, flow, box):
        im_width = image.shape[1]
        # Flip image
        if np.random.randint(2) == 1:
            image = image[:, ::-1, :]
            depth = depth[:, ::-1]
            flow = flow[:, ::-1, :]
            if box.shape[0] > 0:
                flip_box = np.zeros_like(box)
                flip_box[:, 0] = im_width - box[:, 2]
                flip_box[:, 1] = box[:, 1]
                flip_box[:, 2] = im_width - box[:, 0]
                flip_box[:, 3] = box[:, 3]
                flip_box[:, 4] = box[:, 4]
                box = flip_box
        return image, depth, flow, box

    def resize_crop_image(self, image, depth, flow, box):
        orig_im_size = self.orig_im_size
        im_height, im_width = image.shape[0], image.shape[1]

        # Randomly select a scale between orig_im_size and original image size
        h_ratio = orig_im_size[0] * 1.05 / im_height
        w_ratio = orig_im_size[1] * 1.05 / im_width
        r_s = np.random.uniform(max(h_ratio, w_ratio), 1)
        # Resize image, depth and flow and box
        new_height, new_width = int(im_height * r_s), int(im_width * r_s)
        new_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        new_depth = cv2.resize(depth, (new_width, new_height), interpolation=cv2.INTER_AREA)
        new_flow = cv2.resize(flow[:, :, 0:2], (new_width, new_height),
                              interpolation=cv2.INTER_AREA)
        new_flow_mask = cv2.resize(flow[:, :, 2], (new_width, new_height),
                                   interpolation=cv2.INTER_NEAREST)
        if box.shape[0] > 0:
            box[:, 0:4] = box[:, 0:4] * r_s

        # Randomly select a location
        y_s = np.random.randint(0, new_height - orig_im_size[0])
        x_s = np.random.randint(0, new_width - orig_im_size[1])
        # Crop image, depth, flow and box
        image = new_image[y_s:y_s + orig_im_size[0], x_s:x_s + orig_im_size[1], :]
        depth = new_depth[y_s:y_s + orig_im_size[0], x_s:x_s + orig_im_size[1]]
        flow = new_flow[y_s:y_s + orig_im_size[0], x_s:x_s + orig_im_size[1], :]
        flow_mask = new_flow_mask[y_s:y_s + orig_im_size[0], x_s:x_s + orig_im_size[1]]
        flow = np.dstack((flow, flow_mask))
        if box.shape[0] > 0:
            new_box = np.zeros_like(box)
            new_box[:, 0] = box[:, 0] - x_s
            new_box[:, 1] = box[:, 1] - y_s
            new_box[:, 2] = box[:, 2] - x_s
            new_box[:, 3] = box[:, 3] - y_s
            new_box[:, 4] = box[:, 4]
            box = new_box
        return image, depth, flow, box

    @staticmethod
    def rescale_box(box, im_height, im_width):
        if box.shape[0] > 0:
            box[:, 0] = box[:, 0] * 1.0 / im_width
            box[:, 1] = box[:, 1] * 1.0 / im_height
            box[:, 2] = box[:, 2] * 1.0 / im_width
            box[:, 3] = box[:, 3] * 1.0 / im_height
        return box

    @staticmethod
    def get_box_center_size(box):
        xb = (box[:, 0] + box[:, 2]) * 1.0 / 2
        yb = (box[:, 1] + box[:, 3]) * 1.0 / 2
        wb = (box[:, 2] - box[:, 0]) * 1.0
        hb = (box[:, 3] - box[:, 1]) * 1.0
        return xb, yb, wb, hb

    def get_one_sample(self, image_name, depth_name, flow_name, box_name):
        batch_size, orig_im_size = 1, self.orig_im_size
        im_heights, im_widths, num_scale = self.im_heights, self.im_widths, self.num_scale
        ou_heights, ou_widths = self.output_heights, self.output_widths
        images = [np.zeros((batch_size, im_heights[i], im_widths[i], 3)) for i in range(num_scale)]
        depths = [np.zeros((batch_size, im_heights[i], im_widths[i], 1)) for i in range(num_scale)]
        flows = [np.zeros((batch_size, im_heights[i], im_widths[i], 3)) for i in range(num_scale)]
        heatmaps = [np.zeros((batch_size, ou_heights[i], ou_widths[i], 1)) for i in
                    range(num_scale)]
        offsets = [np.zeros((batch_size, ou_heights[i], ou_widths[i], 4)) for i in range(num_scale)]

        orig_image = np.zeros((batch_size, orig_im_size[0], orig_im_size[1], 3))
        orig_depth = np.zeros((batch_size, orig_im_size[0], orig_im_size[1], 1))
        orig_flow = np.zeros((batch_size, orig_im_size[0], orig_im_size[1], 3))
        boxes = []

        for n in range(batch_size):
            image = np.array(Image.open(image_name))
            image = image / 255.0
            depth = flowlib.read_disp_png(depth_name)
            flow = flowlib.read_flow_png(flow_name)
            box = read_box(box_name)

            if self.data_augment:
                image, depth, flow, box = self.flip_image(image, depth, flow, box)
                image, depth, flow, box = self.resize_crop_image(image, depth, flow, box)

            for i in range(num_scale):
                images[i][n, :, :, :] = cv2.resize(image, (im_widths[i], im_heights[i]),
                                                   interpolation=cv2.INTER_AREA)
                depths[i][n, :, :, 0] = cv2.resize(depth, (im_widths[i], im_heights[i]),
                                                   interpolation=cv2.INTER_AREA)
                flows[i][n, :, :, 0:2] = cv2.resize(flow[:, :, 0:2], (im_widths[i], im_heights[i]),
                                                    interpolation=cv2.INTER_AREA)
                flows[i][n, :, :, 2] = cv2.resize(flow[:, :, 2], (im_widths[i], im_heights[i]),
                                                  interpolation=cv2.INTER_NEAREST)
            im_height, im_width = image.shape[0], image.shape[1]
            box = self.rescale_box(box, im_height, im_width)
            boxes.append(box)
            xb, yb, wb, hb = self.get_box_center_size(box)

            for k in range(box.shape[0]):
                for i in range(num_scale):
                    if wb[k] < 2.0 / ou_widths[i] or hb[k] < 2.0 / ou_heights[i]:
                        continue

                    x = np.int(np.floor(xb[k] * ou_widths[i]))
                    y = np.int(np.floor(yb[k] * ou_heights[i]))
                    if x < 0 or x >= ou_widths[i] or y < 0 or y >= ou_heights[i]:
                        continue
                    heatmaps[i][n, y, x, 0] = 1  # Only works for car detection
                    offsets[i][n, y, x, 0] = box[k, 0] - x * 1.0 / ou_widths[i]
                    offsets[i][n, y, x, 1] = box[k, 1] - y * 1.0 / ou_heights[i]
                    offsets[i][n, y, x, 2] = box[k, 2] - x * 1.0 / ou_widths[i]
                    offsets[i][n, y, x, 3] = box[k, 3] - y * 1.0 / ou_heights[i]

                    x = np.int(np.floor(xb[k] * ou_widths[i]))
                    y = np.int(np.ceil(yb[k] * ou_heights[i]))
                    if x < 0 or x >= ou_widths[i] or y < 0 or y >= ou_heights[i]:
                        continue
                    heatmaps[i][n, y, x, 0] = 1  # Only works for car detection
                    offsets[i][n, y, x, 0] = box[k, 0] - x * 1.0 / ou_widths[i]
                    offsets[i][n, y, x, 1] = box[k, 1] - y * 1.0 / ou_heights[i]
                    offsets[i][n, y, x, 2] = box[k, 2] - x * 1.0 / ou_widths[i]
                    offsets[i][n, y, x, 3] = box[k, 3] - y * 1.0 / ou_heights[i]

                    x = np.int(np.ceil(xb[k] * ou_widths[i]))
                    y = np.int(np.floor(yb[k] * ou_heights[i]))
                    if x < 0 or x >= ou_widths[i] or y < 0 or y >= ou_heights[i]:
                        continue
                    heatmaps[i][n, y, x, 0] = 1  # Only works for car detection
                    offsets[i][n, y, x, 0] = box[k, 0] - x * 1.0 / ou_widths[i]
                    offsets[i][n, y, x, 1] = box[k, 1] - y * 1.0 / ou_heights[i]
                    offsets[i][n, y, x, 2] = box[k, 2] - x * 1.0 / ou_widths[i]
                    offsets[i][n, y, x, 3] = box[k, 3] - y * 1.0 / ou_heights[i]

                    x = np.int(np.ceil(xb[k] * ou_widths[i]))
                    y = np.int(np.ceil(yb[k] * ou_heights[i]))
                    if x < 0 or x >= ou_widths[i] or y < 0 or y >= ou_heights[i]:
                        continue
                    heatmaps[i][n, y, x, 0] = 1  # Only works for car detection
                    offsets[i][n, y, x, 0] = box[k, 0] - x * 1.0 / ou_widths[i]
                    offsets[i][n, y, x, 1] = box[k, 1] - y * 1.0 / ou_heights[i]
                    offsets[i][n, y, x, 2] = box[k, 2] - x * 1.0 / ou_widths[i]
                    offsets[i][n, y, x, 3] = box[k, 3] - y * 1.0 / ou_heights[i]

            if self.data_augment:
                orig_image[n, :, :, :] = image
                orig_depth[n, :, :, 0] = depth
                orig_flow[n, :, :, :] = flow
            else:
                orig_image[n, :, :, :] = cv2.resize(image, (orig_im_size[1], orig_im_size[0]),
                                                    interpolation=cv2.INTER_AREA)
                orig_depth[n, :, :, 0] = cv2.resize(depth, (orig_im_size[1], orig_im_size[0]),
                                                    interpolation=cv2.INTER_AREA)
                orig_flow[n, :, :, 0:2] = cv2.resize(flow[:, :, 0:2], (orig_im_size[1], orig_im_size[0]),
                                                     interpolation=cv2.INTER_AREA)
                orig_flow[n, :, :, 2] = cv2.resize(flow[:, :, 2], (orig_im_size[1], orig_im_size[0]),
                                                   interpolation=cv2.INTER_NEAREST)
        sample = {'images': images, 'flows': flows, 'depths': depths, 'heatmaps': heatmaps,
                  'offsets': offsets, 'orig_image': orig_image, 'orig_depth': orig_depth,
                  'orig_flow': orig_flow, 'boxes': boxes}
        return sample
