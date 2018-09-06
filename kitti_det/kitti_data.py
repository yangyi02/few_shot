import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import cv2
from scipy.stats.kde import gaussian_kde
import flowlib

import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


class KittiData(object):
    def __init__(self, data_path, batch_size=8, image_heights=[128], image_widths=[384],
                 output_heights=[16], output_widths=[48], num_scale=1,
                 train_proportion=1, test_proportion=1,
                 show_statistics=''):
        self.name = 'kitti'
        self.image_dir = os.path.join(data_path, 'image_2')
        self.depth_dir = os.path.join(data_path, 'disp_unsup')
        self.flow_dir = os.path.join(data_path, 'flow_unsup')
        self.box_dir = os.path.join(data_path, 'label_2')

        self.batch_size = batch_size
        self.im_heights = image_heights
        self.im_widths = image_widths
        self.num_scale = num_scale
        self.output_heights = output_heights
        self.output_widths = output_widths
        self.orig_im_size = [320, 960]
        self.train_proportion = train_proportion
        self.test_proportion = test_proportion
        self.train_test_split = 0.8

        self.class_map = {'Background': 0, 'Car': 1, 'Van': 2, 'Truck': 3, 'Pedestrian': 4, 'Person_sitting': 5,
                          'Cyclist': 6, 'Tram': 7,
                          'Misc': 8, 'DontCare': 9}
        self.inverse_class_map = {0: 'Background', 1: 'Car', 2: 'Van', 3: 'Truck', 4: 'Pedestrian',
                                  5: 'Person_sitting',
                                  6: 'Cyclist', 7: 'Tram', 8: 'Misc', 9: 'DontCare'}

        self.meta = self.get_meta()
        self.train_meta, self.test_meta = self.split_train_test()
        logging.info('number of training image: %d, number of testing image: %d',
                     len(self.train_meta['img']), len(self.test_meta['img']))
        self.train_anno = self.rearrange_annotation(self.train_meta)
        self.test_anno = self.rearrange_annotation(self.test_meta)
        logging.info('number of training instance: %d, number of testing instance: %d',
                     len(self.train_anno['img']), len(self.test_anno['img']))

        self.show_statistics = show_statistics
        if show_statistics == 'basic':
            self.show_basic_statistics('train')
            self.show_basic_statistics('test')
        elif show_statistics == 'full':
            self.show_full_statistics('train')
            self.show_full_statistics('test')

    def get_meta(self):
        meta = {'img': [], 'depth': [], 'flow': [], 'box': []}
        box_files = os.listdir(os.path.join(self.box_dir))
        # box_files.sort(key=lambda f: int(filter(str.isdigit, f)))
        box_files = sorted(box_files)
        box_file_names = [os.path.join(self.box_dir, f) for f in box_files]
        img_files, depth_files, flow_files = [], [], []
        for f in box_files:
            file_name, file_ext = os.path.splitext(f)
            img_files.append(file_name + '.png')
            depth_files.append(file_name + '.png')
            flow_files.append(file_name + '.png')
        img_file_names = [os.path.join(self.image_dir, f) for f in img_files]
        depth_file_names = [os.path.join(self.depth_dir, f) for f in depth_files]
        flow_file_names = [os.path.join(self.flow_dir, f) for f in flow_files]
        meta['img'].extend(img_file_names)
        meta['depth'].extend(depth_file_names)
        meta['flow'].extend(flow_file_names)
        meta['box'].extend(box_file_names)
        return meta

    def split_train_test(self):
        num_image, thresh = len(self.meta['img']), self.train_test_split
        train_meta = {'img': self.meta['img'][0:int(num_image * thresh)],
                      'depth': self.meta['depth'][0:int(num_image * thresh)],
                      'flow': self.meta['flow'][0:int(num_image * thresh)],
                      'box': self.meta['box'][0:int(num_image * thresh)]}
        test_meta = {'img': self.meta['img'][int(num_image * thresh):],
                     'depth': self.meta['depth'][int(num_image * thresh):],
                     'flow': self.meta['flow'][int(num_image * thresh):],
                     'box': self.meta['box'][int(num_image * thresh):]}
        num_train_image = len(train_meta['img'])
        train_meta['img'] = train_meta['img'][0:int(num_train_image * self.train_proportion)]
        train_meta['depth'] = train_meta['depth'][0:int(num_train_image * self.train_proportion)]
        train_meta['flow'] = train_meta['flow'][0:int(num_train_image * self.train_proportion)]
        train_meta['box'] = train_meta['box'][0:int(num_train_image * self.train_proportion)]
        num_test_image = len(test_meta['img'])
        test_meta['img'] = test_meta['img'][0:int(num_test_image * self.test_proportion)]
        test_meta['depth'] = test_meta['depth'][0:int(num_test_image * self.test_proportion)]
        test_meta['flow'] = test_meta['flow'][0:int(num_test_image * self.test_proportion)]
        test_meta['box'] = test_meta['box'][0:int(num_test_image * self.test_proportion)]
        return train_meta, test_meta

    def rearrange_annotation(self, meta):
        anno = {'img': [], 'depth': [], 'flow': [], 'box': [], 'label': [], 'truncated': [],
                'occluded': []}
        for i in range(len(meta['box'])):
            box_and_label = []
            with open(meta['box'][i]) as txt_file:
                box_info = txt_file.readlines()
            for row in box_info:
                row = row.strip().split(' ')
                if not row[0] in ['Car']:
                    continue
                row[0] = self.class_map[row[0]]
                box_and_label.append(row)
            box_and_label = np.array(box_and_label).astype(np.float)
            anno['img'].append(meta['img'][i])
            anno['depth'].append(meta['depth'][i])
            anno['flow'].append(meta['flow'][i])
            if box_and_label.shape[0] == 0:
                anno['box'].append([])
                anno['label'].append([])
                anno['truncated'].append([])
                anno['occluded'].append([])
            else:
                anno['box'].append(box_and_label[:, 4:8])
                anno['label'].append(box_and_label[:, 0])
                anno['truncated'].append(box_and_label[:, 1])
                anno['occluded'].append(box_and_label[:, 2])
        return anno

    def show_basic_statistics(self, status='train'):
        if status == 'train':
            anno = self.train_anno
        elif status == 'test':
            anno = self.test_anno
        else:
            logging.error('Error: wrong status')
        label_keys = self.inverse_class_map.keys()
        count = dict()
        labels = np.concatenate(anno['label'])
        max_count = 0
        total_count = 0
        for label in label_keys:
            count[label] = (labels == label).sum()
            if count[label] > max_count:
                max_count = count[label]
            total_count = total_count + count[label]
        print(status, count, total_count, max_count * 1.0 / total_count)

    def show_full_statistics(self, status='train'):
        self.show_basic_statistics(status)
        if status == 'train':
            anno = self.train_anno
        elif status == 'test':
            anno = self.test_anno
        else:
            logging.error('Error: wrong status')
        non_empty_box = []
        for box in anno['box']:
            if len(box) > 0:
                non_empty_box.append(box)
        box = np.concatenate(non_empty_box, axis=0)
        x = (box[:, 0] + box[:, 2]) / 2.0
        y = (box[:, 1] + box[:, 3]) / 2.0
        w = (box[:, 2] - box[:, 0]) * 1.0
        h = (box[:, 3] - box[:, 1]) * 1.0
        print(np.min(x), np.max(x), np.mean(x), np.median(x))
        print(np.min(y), np.max(y), np.mean(y), np.median(y))
        print(np.min(w), np.max(w), np.mean(w), np.median(w))
        print(np.min(h), np.max(h), np.mean(h), np.median(h))
        fig, ax = plt.subplots(1)
        ax.scatter(x, y)
        plt.show()
        fig, ax = plt.subplots(1)
        ax.scatter(w, h)
        plt.show()

        num_bins = 100
        fig, ax = plt.subplots(1)
        counts, bin_edges = np.histogram(x, bins=num_bins, normed=True)
        cdf = np.cumsum(counts)
        ax.plot(bin_edges[1:], cdf)
        plt.show()
        counts, bin_edges = np.histogram(y, bins=num_bins, normed=True)
        cdf = np.cumsum(counts)
        fig, ax = plt.subplots(1)
        ax.plot(bin_edges[1:], cdf)
        plt.show()
        counts, bin_edges = np.histogram(w, bins=num_bins, normed=True)
        cdf = np.cumsum(counts)
        fig, ax = plt.subplots(1)
        ax.plot(bin_edges[1:], cdf)
        plt.show()
        counts, bin_edges = np.histogram(h, bins=num_bins, normed=True)
        cdf = np.cumsum(counts)
        fig, ax = plt.subplots(1)
        ax.plot(bin_edges[1:], cdf)
        plt.show()

        # fig, ax = plt.subplots(1)
        # k = gaussian_kde(np.vstack([x, y]))
        # xi, yi = np.mgrid[0:1:0.01, 0:1:0.01]
        # zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        # # ax.pcolormesh(xi, yi, zi.reshape(xi.shape))
        # ax.contourf(xi, yi, zi.reshape(xi.shape))
        # plt.show()
        # fig, ax = plt.subplots(1)
        # k = gaussian_kde(np.vstack([w, h]))
        # xi, yi = np.mgrid[0:1:0.01, 0:1:0.01]
        # zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        # # ax.pcolormesh(xi, yi, zi.reshape(xi.shape))
        # ax.contourf(xi, yi, zi.reshape(xi.shape))
        # plt.show()

    def data_augmentation(self, image, depth, flow, box):
        im_width = image.shape[1]
        # Flip image
        if np.random.randint(2) == 1:
            image = image[:, ::-1, :]
            depth = depth[:, ::-1]
            flow = flow[:, ::-1, :]
            if len(box) > 0 and len(box.shape) == 1:
                flip_box = np.zeros(4)
                flip_box[0] = im_width - box[2]
                flip_box[1] = box[1]
                flip_box[2] = im_width - box[0]
                flip_box[3] = box[3]
                box = flip_box
            elif len(box) > 0 and len(box.shape) == 2:
                flip_box = np.zeros_like(box)
                flip_box[:, 0] = im_width - box[:, 2]
                flip_box[:, 1] = box[:, 1]
                flip_box[:, 2] = im_width - box[:, 0]
                flip_box[:, 3] = box[:, 3]
                box = flip_box
        return image, depth, flow, box

    def crop_image(self, image, depth, flow, box):
        im_height, im_width = image.shape[0], image.shape[1]

        # Randomly select a scale
        h_ratio = self.orig_im_size[0] * 1.05 / im_height
        w_ratio = self.orig_im_size[1] * 1.05 / im_width
        r_s = np.random.uniform(max(h_ratio, w_ratio), 1)

        # Resize image, depth and flow
        new_height, new_width = int(im_height * r_s), int(im_width * r_s)
        new_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        new_depth = cv2.resize(depth, (new_width, new_height), interpolation=cv2.INTER_AREA)
        new_flow = cv2.resize(flow[:, :, 0:2], (new_width, new_height),
                              interpolation=cv2.INTER_AREA)
        new_flow_mask = cv2.resize(flow[:, :, 2], (new_width, new_height),
                                   interpolation=cv2.INTER_NEAREST)

        # Randomly select a location
        y_s = np.random.randint(0, new_height - self.orig_im_size[0])
        x_s = np.random.randint(0, new_width - self.orig_im_size[1])
        image = new_image[y_s:y_s + self.orig_im_size[0], x_s:x_s + self.orig_im_size[1], :]
        depth = new_depth[y_s:y_s + self.orig_im_size[0], x_s:x_s + self.orig_im_size[1]]
        flow = new_flow[y_s:y_s + self.orig_im_size[0], x_s:x_s + self.orig_im_size[1], :]
        flow_mask = new_flow_mask[y_s:y_s + self.orig_im_size[0], x_s:x_s + self.orig_im_size[1]]
        flow = np.dstack((flow, flow_mask))
        if len(box) > 0 and len(box.shape) == 1:
            new_box = np.zeros(4)
            new_box[0] = box[0] * r_s - x_s
            new_box[1] = box[1] * r_s - y_s
            new_box[2] = box[2] * r_s - x_s
            new_box[3] = box[3] * r_s - y_s
            box = new_box
        elif len(box) > 0 and len(box.shape) == 2:
            new_box = np.zeros_like(box)
            new_box[:, 0] = box[:, 0] * r_s - x_s
            new_box[:, 1] = box[:, 1] * r_s - y_s
            new_box[:, 2] = box[:, 2] * r_s - x_s
            new_box[:, 3] = box[:, 3] * r_s - y_s
            box = new_box
        return image, depth, flow, box

    def get_next_batch(self, status='train', cnt=0, index=None):
        if status == 'train':
            anno = self.train_anno
        elif status == 'test':
            anno = self.test_anno
        else:
            logging.error('Error: wrong status')
        if index is None:
            index = np.arange(len(anno['img']))
        batch_size, orig_im_size = self.batch_size, self.orig_im_size
        im_heights, im_widths, num_scale = self.im_heights, self.im_widths, self.num_scale
        ou_heights, ou_widths = self.output_heights, self.output_widths
        images = [np.zeros((batch_size, im_heights[i], im_widths[i], 3)) for i in range(num_scale)]
        orig_image = np.zeros((batch_size, orig_im_size[0], orig_im_size[1], 3))
        depths = [np.zeros((batch_size, im_heights[i], im_widths[i], 1)) for i in range(num_scale)]
        orig_depth = np.zeros((batch_size, orig_im_size[0], orig_im_size[1], 1))
        flows = [np.zeros((batch_size, im_heights[i], im_widths[i], 3)) for i in range(num_scale)]
        orig_flow = np.zeros((batch_size, orig_im_size[0], orig_im_size[1], 3))
        boxes = []
        label_maps = [np.zeros((batch_size, ou_heights[i], ou_widths[i], 1))
                      for i in range(num_scale)]
        offsets = [np.zeros((batch_size, ou_heights[i], ou_widths[i], 4))
                       for i in range(num_scale)]
        restart = False
        for n in range(batch_size):
            if cnt >= len(index):
                cnt = 0
                restart = True
            image = np.array(Image.open(anno['img'][index[cnt]]))
            image = image / 255.0
            depth = flowlib.read_disp_png(anno['depth'][index[cnt]])
            flow = flowlib.read_flow_png(anno['flow'][index[cnt]])
            box = np.array(anno['box'][index[cnt]])
            label = anno['label'][index[cnt]]

            if status == 'train':
                image, depth, flow, box = self.data_augmentation(image, depth, flow, box)
                image, depth, flow, box = self.crop_image(image, depth, flow, box)
            im_height, im_width = image.shape[0], image.shape[1]

            for i in range(num_scale):
                images[i][n, :, :, :] = cv2.resize(image, (im_widths[i], im_heights[i]),
                                                   interpolation=cv2.INTER_AREA)
                depths[i][n, :, :, 0] = cv2.resize(depth, (im_widths[i], im_heights[i]),
                                                   interpolation=cv2.INTER_AREA)
                flows[i][n, :, :, 0:2] = cv2.resize(flow[:, :, 0:2], (im_widths[i], im_heights[i]),
                                                    interpolation=cv2.INTER_AREA)
                flows[i][n, :, :, 2] = cv2.resize(flow[:, :, 2], (im_widths[i], im_heights[i]),
                                                  interpolation=cv2.INTER_NEAREST)
            if status == 'train':
                orig_image[n, :, :, :] = image
                orig_depth[n, :, :, 0] = depth
                orig_flow[n, :, :, :] = flow
            else:
                orig_image[n, :, :, :] = cv2.resize(image, (self.orig_im_size[1], self.orig_im_size[0]),
                                                    interpolation=cv2.INTER_AREA)
                orig_depth[n, :, :, 0] = cv2.resize(depth, (self.orig_im_size[1], self.orig_im_size[0]),
                                                    interpolation=cv2.INTER_AREA)
                orig_flow[n, :, :, 0:2] = cv2.resize(flow[:, :, 0:2], (self.orig_im_size[1], self.orig_im_size[0]),
                                                     interpolation=cv2.INTER_AREA)
                orig_flow[n, :, :, 2] = cv2.resize(flow[:, :, 2], (self.orig_im_size[1], self.orig_im_size[0]),
                                                   interpolation=cv2.INTER_NEAREST)
            if len(box) > 0:
                box[:, 0] = box[:, 0] * 1.0 / im_width
                box[:, 1] = box[:, 1] * 1.0 / im_height
                box[:, 2] = box[:, 2] * 1.0 / im_width
                box[:, 3] = box[:, 3] * 1.0 / im_height
                boxes.append(box)

                x = (box[:, 0] + box[:, 2]) * 1.0 / 2
                y = (box[:, 1] + box[:, 3]) * 1.0 / 2
                w = (box[:, 2] - box[:, 0]) * 1.0
                h = (box[:, 3] - box[:, 1]) * 1.0
                for k in range(box.shape[0]):
                    for i in range(num_scale):
                        # if w[k] < 1.0 / ou_widths[i] or h[k] < 1.0 / ou_heights[i]:
                        #     continue
                        # x_c = np.int(np.round(x[k] * ou_widths[i]))
                        # y_c = np.int(np.round(y[k] * ou_heights[i]))
                        # if x_c < 0 or x_c >= self.output_widths[i] or y_c < 0 or y_c >= self.output_heights[i]:
                        #     continue
                        # label_maps[i][n, y_c, x_c, 0] = 1  # Only works for car detection
                        # # x1 = max(x_c - 1, 0)
                        # # x2 = min(x_c + 1, self.output_widths[i]-1)
                        # # y1 = max(y_c - 1, 0)
                        # # y2 = min(y_c + 1, self.output_heights[i]-1)
                        # # label_maps[i][n, y1:y2, x1:x2, 0] = 1  # Only works for car detection
                        # offsets[i][n, y_c, x_c, 0] = box[k, 0] - x_c * 1.0 / ou_widths[i]
                        # offsets[i][n, y_c, x_c, 1] = box[k, 1] - y_c * 1.0 / ou_heights[i]
                        # offsets[i][n, y_c, x_c, 2] = box[k, 2] - x_c * 1.0 / ou_widths[i]
                        # offsets[i][n, y_c, x_c, 3] = box[k, 3] - y_c * 1.0 / ou_heights[i]

                        if w[k] < 2.0 / ou_widths[i] or h[k] < 2.0 / ou_heights[i]:
                            continue

                        x_c = np.int(np.floor(x[k] * ou_widths[i]))
                        y_c = np.int(np.floor(y[k] * ou_heights[i]))
                        if x_c < 0 or x_c >= self.output_widths[i] or y_c < 0 or y_c >= self.output_heights[i]:
                            continue
                        label_maps[i][n, y_c, x_c, 0] = 1  # Only works for car detection
                        offsets[i][n, y_c, x_c, 0] = box[k, 0] - x_c * 1.0 / ou_widths[i]
                        offsets[i][n, y_c, x_c, 1] = box[k, 1] - y_c * 1.0 / ou_heights[i]
                        offsets[i][n, y_c, x_c, 2] = box[k, 2] - x_c * 1.0 / ou_widths[i]
                        offsets[i][n, y_c, x_c, 3] = box[k, 3] - y_c * 1.0 / ou_heights[i]

                        x_c = np.int(np.floor(x[k] * ou_widths[i]))
                        y_c = np.int(np.ceil(y[k] * ou_heights[i]))
                        if x_c < 0 or x_c >= self.output_widths[i] or y_c < 0 or y_c >= self.output_heights[i]:
                            continue
                        label_maps[i][n, y_c, x_c, 0] = 1  # Only works for car detection
                        offsets[i][n, y_c, x_c, 0] = box[k, 0] - x_c * 1.0 / ou_widths[i]
                        offsets[i][n, y_c, x_c, 1] = box[k, 1] - y_c * 1.0 / ou_heights[i]
                        offsets[i][n, y_c, x_c, 2] = box[k, 2] - x_c * 1.0 / ou_widths[i]
                        offsets[i][n, y_c, x_c, 3] = box[k, 3] - y_c * 1.0 / ou_heights[i]

                        x_c = np.int(np.ceil(x[k] * ou_widths[i]))
                        y_c = np.int(np.floor(y[k] * ou_heights[i]))
                        if x_c < 0 or x_c >= self.output_widths[i] or y_c < 0 or y_c >= self.output_heights[i]:
                            continue
                        label_maps[i][n, y_c, x_c, 0] = 1  # Only works for car detection
                        offsets[i][n, y_c, x_c, 0] = box[k, 0] - x_c * 1.0 / ou_widths[i]
                        offsets[i][n, y_c, x_c, 1] = box[k, 1] - y_c * 1.0 / ou_heights[i]
                        offsets[i][n, y_c, x_c, 2] = box[k, 2] - x_c * 1.0 / ou_widths[i]
                        offsets[i][n, y_c, x_c, 3] = box[k, 3] - y_c * 1.0 / ou_heights[i]

                        x_c = np.int(np.ceil(x[k] * ou_widths[i]))
                        y_c = np.int(np.ceil(y[k] * ou_heights[i]))
                        if x_c < 0 or x_c >= self.output_widths[i] or y_c < 0 or y_c >= self.output_heights[i]:
                            continue
                        label_maps[i][n, y_c, x_c, 0] = 1  # Only works for car detection
                        offsets[i][n, y_c, x_c, 0] = box[k, 0] - x_c * 1.0 / ou_widths[i]
                        offsets[i][n, y_c, x_c, 1] = box[k, 1] - y_c * 1.0 / ou_heights[i]
                        offsets[i][n, y_c, x_c, 2] = box[k, 2] - x_c * 1.0 / ou_widths[i]
                        offsets[i][n, y_c, x_c, 3] = box[k, 3] - y_c * 1.0 / ou_heights[i]
            else:
                boxes.append([])
            cnt = cnt + 1
        for i in range(num_scale):
            images[i] = images[i].transpose((0, 3, 1, 2))
            depths[i] = depths[i].transpose((0, 3, 1, 2))
            flows[i] = flows[i].transpose((0, 3, 1, 2))
            label_maps[i] = label_maps[i].transpose((0, 3, 1, 2))
            offsets[i] = offsets[i].transpose((0, 3, 1, 2))
        orig_image = orig_image.transpose((0, 3, 1, 2))
        orig_depth = orig_depth.transpose((0, 3, 1, 2))
        orig_flow = orig_flow.transpose((0, 3, 1, 2))
        return images, orig_image, depths, orig_depth, flows, orig_flow, boxes, \
            label_maps, offsets, cnt, restart

    def get_one_sample(self, image_name, depth_name, flow_name, box_name):
        image = np.array(Image.open(image_name))
        image = image / 255.0
        depth = flowlib.read_disp_png(depth_name)
        flow = flowlib.read_flow_png(flow_name)

        box_and_label = []
        with open(box_name) as txt_file:
            box_info = txt_file.readlines()
        for row in box_info:
            row = row.strip().split(' ')
            if not row[0] in ['Car']:
                continue
            row[0] = self.class_map[row[0]]
            box_and_label.append(row)
        box_and_label = np.array(box_and_label).astype(np.float)
        if box_and_label.shape[0] == 0:
            box = []
            label = []
        else:
            box = box_and_label[:, 4:8]
            label = box_and_label[:, 0]

        batch_size = 1
        orig_im_size = self.orig_im_size
        im_heights, im_widths, num_scale = self.im_heights, self.im_widths, self.num_scale
        ou_heights, ou_widths = self.output_heights, self.output_widths
        images = [np.zeros((batch_size, im_heights[i], im_widths[i], 3)) for i in range(num_scale)]
        orig_image = np.zeros((batch_size, orig_im_size[0], orig_im_size[1], 3))
        depths = [np.zeros((batch_size, im_heights[i], im_widths[i], 1)) for i in range(num_scale)]
        orig_depth = np.zeros((batch_size, orig_im_size[0], orig_im_size[1], 1))
        flows = [np.zeros((batch_size, im_heights[i], im_widths[i], 3)) for i in range(num_scale)]
        orig_flow = np.zeros((batch_size, orig_im_size[0], orig_im_size[1], 3))
        boxes = []
        label_maps = [np.zeros((batch_size, ou_heights[i], ou_widths[i], 1))
                      for i in range(num_scale)]
        offsets = [np.zeros((batch_size, ou_heights[i], ou_widths[i], 4))
                       for i in range(num_scale)]

        # image, depth, flow, box = self.crop_image(image, depth, flow, box)
        im_height, im_width = image.shape[0], image.shape[1]

        for i in range(num_scale):
            images[i][0, :, :, :] = cv2.resize(image, (im_widths[i], im_heights[i]),
                                               interpolation=cv2.INTER_AREA)
            depths[i][0, :, :, 0] = cv2.resize(depth, (im_widths[i], im_heights[i]),
                                               interpolation=cv2.INTER_AREA)
            flows[i][0, :, :, 0:2] = cv2.resize(flow[:, :, 0:2], (im_widths[i], im_heights[i]),
                                                interpolation=cv2.INTER_AREA)
            flows[i][0, :, :, 2] = cv2.resize(flow[:, :, 2], (im_widths[i], im_heights[i]),
                                              interpolation=cv2.INTER_NEAREST)
        orig_image[0, :, :, :] = cv2.resize(image, (self.orig_im_size[1], self.orig_im_size[0]),
                                            interpolation=cv2.INTER_AREA)
        orig_depth[0, :, :, 0] = cv2.resize(depth, (self.orig_im_size[1], self.orig_im_size[0]),
                                            interpolation=cv2.INTER_AREA)
        orig_flow[0, :, :, 0:2] = cv2.resize(flow[:, :, 0:2], (self.orig_im_size[1], self.orig_im_size[0]),
                                             interpolation=cv2.INTER_AREA)
        orig_flow[0, :, :, 2] = cv2.resize(flow[:, :, 2], (self.orig_im_size[1], self.orig_im_size[0]),
                                           interpolation=cv2.INTER_NEAREST)
        if len(box) > 0:
            box[:, 0] = box[:, 0] * 1.0 / im_width
            box[:, 1] = box[:, 1] * 1.0 / im_height
            box[:, 2] = box[:, 2] * 1.0 / im_width
            box[:, 3] = box[:, 3] * 1.0 / im_height
            boxes.append(box)
            x = (box[:, 0] + box[:, 2]) * 1.0 / 2
            y = (box[:, 1] + box[:, 3]) * 1.0 / 2
            w = (box[:, 2] - box[:, 0]) * 1.0
            h = (box[:, 3] - box[:, 1]) * 1.0
            for k in range(box.shape[0]):
                for i in range(num_scale):
                    # if w[k] < 1.0 / ou_widths[i] or h[k] < 1.0 / ou_heights[i]:
                    #     continue
                    # x_c = np.int(np.round(x[k] * ou_widths[i]))
                    # y_c = np.int(np.round(y[k] * ou_heights[i]))
                    # label_maps[i][0, y_c, x_c, 0] = 1  # Only works for car detection
                    # # x1 = max(x_c - 1, 0)
                    # # x2 = min(x_c + 1, self.output_widths[i]-1)
                    # # y1 = max(y_c - 1, 0)
                    # # y2 = min(y_c + 1, self.output_heights[i]-1)
                    # # label_maps[i][0, y1:y2, x1:x2, 0] = 1  # Only works for car detection
                    # # offsets[i][0, y_c, x_c, 0] = w[k]
                    # # offsets[i][0, y_c, x_c, 1] = h[k]
                    # offsets[i][0, y_c, x_c, 0] = box[k, 0] - x_c * 1.0 / ou_widths[i]
                    # offsets[i][0, y_c, x_c, 1] = box[k, 1] - y_c * 1.0 / ou_heights[i]
                    # offsets[i][0, y_c, x_c, 2] = box[k, 2] - x_c * 1.0 / ou_widths[i]
                    # offsets[i][0, y_c, x_c, 3] = box[k, 3] - y_c * 1.0 / ou_heights[i]

                    if w[k] < 2.0 / ou_widths[i] or h[k] < 2.0 / ou_heights[i]:
                        continue

                    x_c = np.int(np.floor(x[k] * ou_widths[i]))
                    y_c = np.int(np.floor(y[k] * ou_heights[i]))
                    if x_c < 0 or x_c >= self.output_widths[i] or y_c < 0 or y_c >= self.output_heights[i]:
                        continue
                    label_maps[i][0, y_c, x_c, 0] = 1  # Only works for car detection
                    offsets[i][0, y_c, x_c, 0] = box[k, 0] - x_c * 1.0 / ou_widths[i]
                    offsets[i][0, y_c, x_c, 1] = box[k, 1] - y_c * 1.0 / ou_heights[i]
                    offsets[i][0, y_c, x_c, 2] = box[k, 2] - x_c * 1.0 / ou_widths[i]
                    offsets[i][0, y_c, x_c, 3] = box[k, 3] - y_c * 1.0 / ou_heights[i]

                    x_c = np.int(np.floor(x[k] * ou_widths[i]))
                    y_c = np.int(np.ceil(y[k] * ou_heights[i]))
                    if x_c < 0 or x_c >= self.output_widths[i] or y_c < 0 or y_c >= self.output_heights[i]:
                        continue
                    label_maps[i][0, y_c, x_c, 0] = 1  # Only works for car detection
                    offsets[i][0, y_c, x_c, 0] = box[k, 0] - x_c * 1.0 / ou_widths[i]
                    offsets[i][0, y_c, x_c, 1] = box[k, 1] - y_c * 1.0 / ou_heights[i]
                    offsets[i][0, y_c, x_c, 2] = box[k, 2] - x_c * 1.0 / ou_widths[i]
                    offsets[i][0, y_c, x_c, 3] = box[k, 3] - y_c * 1.0 / ou_heights[i]

                    x_c = np.int(np.ceil(x[k] * ou_widths[i]))
                    y_c = np.int(np.floor(y[k] * ou_heights[i]))
                    if x_c < 0 or x_c >= self.output_widths[i] or y_c < 0 or y_c >= self.output_heights[i]:
                        continue
                    label_maps[i][0, y_c, x_c, 0] = 1  # Only works for car detection
                    offsets[i][0, y_c, x_c, 0] = box[k, 0] - x_c * 1.0 / ou_widths[i]
                    offsets[i][0, y_c, x_c, 1] = box[k, 1] - y_c * 1.0 / ou_heights[i]
                    offsets[i][0, y_c, x_c, 2] = box[k, 2] - x_c * 1.0 / ou_widths[i]
                    offsets[i][0, y_c, x_c, 3] = box[k, 3] - y_c * 1.0 / ou_heights[i]

                    x_c = np.int(np.ceil(x[k] * ou_widths[i]))
                    y_c = np.int(np.ceil(y[k] * ou_heights[i]))
                    if x_c < 0 or x_c >= self.output_widths[i] or y_c < 0 or y_c >= self.output_heights[i]:
                        continue
                    label_maps[i][0, y_c, x_c, 0] = 1  # Only works for car detection
                    offsets[i][0, y_c, x_c, 0] = box[k, 0] - x_c * 1.0 / ou_widths[i]
                    offsets[i][0, y_c, x_c, 1] = box[k, 1] - y_c * 1.0 / ou_heights[i]
                    offsets[i][0, y_c, x_c, 2] = box[k, 2] - x_c * 1.0 / ou_widths[i]
                    offsets[i][0, y_c, x_c, 3] = box[k, 3] - y_c * 1.0 / ou_heights[i]
        else:
            boxes.append([])
        labels = label
        for i in range(num_scale):
            images[i] = images[i].transpose((0, 3, 1, 2))
            depths[i] = depths[i].transpose((0, 3, 1, 2))
            flows[i] = flows[i].transpose((0, 3, 1, 2))
            label_maps[i] = label_maps[i].transpose((0, 3, 1, 2))
            offsets[i] = offsets[i].transpose((0, 3, 1, 2))
        orig_image = orig_image.transpose((0, 3, 1, 2))
        orig_depth = orig_depth.transpose((0, 3, 1, 2))
        orig_flow = orig_flow.transpose((0, 3, 1, 2))
        return images, orig_image, depths, orig_depth, flows, orig_flow, boxes, \
            label_maps, offsets

    def visualize(self, images, orig_image, depths, orig_depth, flows, orig_flow, boxes,
                  label_maps, offsets, idx=None):
        if idx is None:
            idx = 0
        # Plot original image and depth with bounding box
        orig_im = orig_image[idx, :, :, :].transpose(1, 2, 0)
        orig_dp = orig_depth[idx, 0, :, :]
        orig_fl = orig_flow[idx, :, :, :].transpose(1, 2, 0)
        if len(boxes[idx]) > 0:
            b = boxes[idx].copy()
            im_height, im_width = orig_im.shape[0], orig_im.shape[1]
            b[:, 0], b[:, 2] = b[:, 0] * im_width, b[:, 2] * im_width
            b[:, 1], b[:, 3] = b[:, 1] * im_height, b[:, 3] * im_height
        else:
            b = []
        fig, ax = plt.subplots(1)
        ax.imshow(orig_im)
        if len(b) > 0:
            for i in range(b.shape[0]):
                rect = patches.Rectangle((b[i][0], b[i][1]), b[i][2] - 1 - b[i][0],
                                         b[i][3] - 1 - b[i][1], linewidth=2,
                                         edgecolor='g', facecolor='none')
                ax.add_patch(rect)
        plt.show()
        fig, ax = plt.subplots(1)
        orig_dp = flowlib.visualize_disp(orig_dp)
        ax.imshow(orig_dp)
        if len(b) > 0:
            for i in range(b.shape[0]):
                rect = patches.Rectangle((b[i][0], b[i][1]), b[i][2] - 1 - b[i][0],
                                         b[i][3] - 1 - b[i][1], linewidth=2,
                                         edgecolor='w', facecolor='none')
                ax.add_patch(rect)
        plt.show()
        fig, ax = plt.subplots(1)
        orig_fl = flowlib.visualize_flow(orig_fl)
        ax.imshow(orig_fl)
        if len(b) > 0:
            for i in range(b.shape[0]):
                rect = patches.Rectangle((b[i][0], b[i][1]), b[i][2] - 1 - b[i][0],
                                         b[i][3] - 1 - b[i][1], linewidth=2,
                                         edgecolor='k', facecolor='none')
                ax.add_patch(rect)
        plt.show()
        # construct a full image containing all scale images
        max_im_size = np.max(np.array(self.im_widths))
        sum_im_size = np.sum(np.array(self.im_heights))
        im_all = np.zeros((sum_im_size, max_im_size, 3))
        dp_all = np.zeros((sum_im_size, max_im_size, 3))
        fl_all = np.zeros((sum_im_size, max_im_size, 3))
        cnt = 0
        for i in range(len(images)):
            im = images[i][idx, :, :, :].transpose(1, 2, 0)
            height, width = im.shape[0], im.shape[1]
            im_all[cnt:cnt + height, 0:width, :] = im
            dp = depths[i][idx, 0, :, :]
            dp = flowlib.visualize_disp(dp)
            dp_all[cnt:cnt + height, 0:width, :] = dp
            fl = flows[i][idx, :, :, :].transpose(1, 2, 0)
            fl = flowlib.visualize_flow(fl)
            fl_all[cnt:cnt + height, 0:width, :] = fl
            cnt = cnt + height
        fig, ax = plt.subplots(1)
        ax.imshow(im_all)
        plt.show()
        fig, ax = plt.subplots(1)
        ax.imshow(dp_all)
        plt.show()
        fig, ax = plt.subplots(1)
        ax.imshow(fl_all.astype(np.uint8))
        plt.show()

        max_ou_size = np.max(np.array(self.output_widths))
        sum_ou_size = np.sum(np.array(self.output_heights))
        lb_all = np.zeros((sum_ou_size, max_ou_size))
        cnt = 0
        for i in range(len(label_maps)):
            lb = label_maps[i][idx, 0, :, :]
            height, width = lb.shape[0], lb.shape[1]
            lb_all[cnt:cnt + height, 0:width] = lb
            cnt = cnt + height
        fig, ax = plt.subplots(1)
        ax.imshow(lb_all)
        plt.show()

        label_on_image = self.visualize_prediction_on_image(im_all, lb_all)
        fig, ax = plt.subplots(1)
        ax.imshow(label_on_image)
        plt.show()

        fig, ax = plt.subplots(1)
        ax.imshow(orig_im)
        for i in range(len(label_maps)):
            of = offsets[i][idx, :, :, :].transpose(1, 2, 0)
            lb = label_maps[i][idx, 0, :, :]
            ou_height, ou_width = lb.shape[0], lb.shape[1]
            b_idx = np.nonzero(lb)
            y_c, x_c = b_idx[0], b_idx[1]
            b = np.zeros((len(b_idx[0]), 4))
            for k in range(b.shape[0]):
                offset = of[y_c[k], x_c[k], :]
                b[k, 0] = x_c[k] * 1.0 / ou_width + offset[0]
                b[k, 1] = y_c[k] * 1.0 / ou_height + offset[1]
                b[k, 2] = x_c[k] * 1.0 / ou_width + offset[2]
                b[k, 3] = y_c[k] * 1.0 / ou_height + offset[3]
            if len(b) > 0:
                b[:, 0], b[:, 2] = b[:, 0] * im_width, b[:, 2] * im_width
                b[:, 1], b[:, 3] = b[:, 1] * im_height, b[:, 3] * im_height
            for k in range(b.shape[0]):
                rect = patches.Rectangle((b[k][0], b[k][1]), b[k][2] - 1 - b[k][0],
                                         b[k][3] - 1 - b[k][1], linewidth=2,
                                         edgecolor='g', facecolor='none')
                ax.add_patch(rect)
        plt.show()

    @staticmethod
    def visualize_prediction_on_image(image, prediction):
        im_height, im_width = image.shape[0], image.shape[1]
        pred = cv2.resize(prediction, (im_width, im_height), interpolation=cv2.INTER_LINEAR)
        pred = np.dstack((pred, pred, pred))
        pred_on_image = image * 0.2 + pred * 0.8
        return pred_on_image

