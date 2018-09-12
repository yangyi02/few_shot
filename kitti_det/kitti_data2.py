import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import cv2
from scipy.stats.kde import gaussian_kde
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import flowlib

import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


class KittiData(object):
    def __init__(self, data_path, train_proportion=1, test_proportion=1, show_statistics=''):
        self.name = 'kitti'
        self.image_dir = os.path.join(data_path, 'image_2')
        self.depth_dir = os.path.join(data_path, 'disp_unsup')
        self.flow_dir = os.path.join(data_path, 'flow_unsup')
        self.box_dir = os.path.join(data_path, 'label_2')

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
                     len(self.train_meta['image']), len(self.test_meta['image']))
        self.train_meta = self.rearrange_annotation(self.train_meta)
        self.test_meta = self.rearrange_annotation(self.test_meta)
        logging.info('number of training instance: %d, number of testing instance: %d',
                     len(self.train_meta['image']), len(self.test_meta['image']))

        self.show_statistics = show_statistics
        if show_statistics == 'basic':
            self.show_basic_statistics('train')
            self.show_basic_statistics('test')
        elif show_statistics == 'full':
            self.show_full_statistics('train')
            self.show_full_statistics('test')

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
        meta['box'] = [os.path.join(self.box_dir, f) for f in box_files]
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
        # Take a proportion of train or test data for ablation usage
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

    def rearrange_annotation(self, meta):
        anno = {'image': [], 'depth': [], 'flow': [], 'box': [], 'heatmap': [], 'truncated': [],
                'occluded': []}
        for i in range(len(meta['image'])):
            box_and_heatmap = []
            with open(meta['box'][i]) as txt_file:
                box_info = txt_file.readlines()
            for row in box_info:
                row = row.strip().split(' ')
                if not row[0] in ['Car']: # Only take car objects right now
                    continue
                row[0] = self.class_map[row[0]]
                box_and_heatmap.append(row)
            box_and_heatmap = np.array(box_and_heatmap).astype(np.float)
            anno['image'].append(meta['image'][i])
            anno['depth'].append(meta['depth'][i])
            anno['flow'].append(meta['flow'][i])
            if box_and_heatmap.shape[0] == 0:
                anno['box'].append([])
                anno['heatmap'].append([])
                anno['truncated'].append([])
                anno['occluded'].append([])
            else:
                anno['box'].append(box_and_heatmap[:, 4:8])
                anno['heatmap'].append(box_and_heatmap[:, 0])
                anno['truncated'].append(box_and_heatmap[:, 1])
                anno['occluded'].append(box_and_heatmap[:, 2])
        return anno

    def show_basic_statistics(self, status='train'):
        if status == 'train':
            anno = self.train_anno
        elif status == 'test':
            anno = self.test_anno
        else:
            logging.error('Error: wrong status')
        heatmap_keys = self.inverse_class_map.keys()
        count = dict()
        heatmaps = np.concatenate(anno['heatmap'])
        max_count = 0
        total_count = 0
        for heatmap in heatmap_keys:
            count[heatmap] = (heatmaps == heatmap).sum()
            if count[heatmap] > max_count:
                max_count = count[heatmap]
            total_count = total_count + count[heatmap]
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


class KittiDataset(Dataset):
    def __init__(self, meta, image_heights, image_widths, num_scale, output_heights, output_widths, data_augment=False, transform=None):
        self.meta = meta
        self.im_heights = image_heights
        self.im_widths = image_widths
        self.num_scale = num_scale
        self.output_heights = output_heights
        self.output_widths = output_widths
        self.orig_im_size = [320, 960]
        self.data_augment = data_augment
        self.transform = transform

    def __len__(self):
        return len(self.meta['image'])

    def __getitem__(self, idx):
        orig_im_size = self.orig_im_size
        im_heights, im_widths, num_scale = self.im_heights, self.im_widths, self.num_scale
        ou_heights, ou_widths = self.output_heights, self.output_widths
        images = [np.zeros((im_heights[i], im_widths[i], 3)) for i in range(num_scale)]
        orig_image = np.zeros((orig_im_size[0], orig_im_size[1], 3))
        depths = [np.zeros((im_heights[i], im_widths[i], 1)) for i in range(num_scale)]
        orig_depth = np.zeros((orig_im_size[0], orig_im_size[1], 1))
        flows = [np.zeros((im_heights[i], im_widths[i], 3)) for i in range(num_scale)]
        orig_flow = np.zeros((orig_im_size[0], orig_im_size[1], 3))
        boxes = []
        heatmaps = [np.zeros((ou_heights[i], ou_widths[i], 1)) for i in range(num_scale)]
        offsets = [np.zeros((ou_heights[i], ou_widths[i], 4)) for i in range(num_scale)]

        image = np.array(Image.open(self.meta['image'][idx]))
        image = image / 255.0
        depth = flowlib.read_disp_png(self.meta['depth'][idx])
        flow = flowlib.read_flow_png(self.meta['flow'][idx])
        box = np.array(self.meta['box'][idx])
        heatmap = self.meta['heatmap'][idx]

        if self.data_augment:
            image, depth, flow, box = self.data_augmentation(image, depth, flow, box)
            image, depth, flow, box = self.crop_image(image, depth, flow, box)
        im_height, im_width = image.shape[0], image.shape[1]

        for i in range(num_scale):
            images[i] = cv2.resize(image, (im_widths[i], im_heights[i]), interpolation=cv2.INTER_AREA)
            depths[i][:, :, 0] = cv2.resize(depth, (im_widths[i], im_heights[i]),
                                            interpolation=cv2.INTER_AREA)
            flows[i][:, :, 0:2] = cv2.resize(flow[:, :, 0:2], (im_widths[i], im_heights[i]),
                                             interpolation=cv2.INTER_AREA)
            flows[i][:, :, 2] = cv2.resize(flow[:, :, 2], (im_widths[i], im_heights[i]),
                                           interpolation=cv2.INTER_NEAREST)
        if self.data_augment:
            orig_image = image
            orig_depth[:, :, 0] = depth
            orig_flow = flow
        else:
            orig_image = cv2.resize(image, (self.orig_im_size[1], self.orig_im_size[0]),
                                    interpolation=cv2.INTER_AREA)
            orig_depth[:, :, 0] = cv2.resize(depth, (self.orig_im_size[1], self.orig_im_size[0]),
                                             interpolation=cv2.INTER_AREA)
            orig_flow[:, :, 0:2] = cv2.resize(flow[:, :, 0:2], (self.orig_im_size[1], self.orig_im_size[0]),
                                              interpolation=cv2.INTER_AREA)
            orig_flow[:, :, 2] = cv2.resize(flow[:, :, 2], (self.orig_im_size[1], self.orig_im_size[0]),
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
                    # heatmaps[i][n, y_c, x_c, 0] = 1  # Only works for car detection
                    # # x1 = max(x_c - 1, 0)
                    # # x2 = min(x_c + 1, self.output_widths[i]-1)
                    # # y1 = max(y_c - 1, 0)
                    # # y2 = min(y_c + 1, self.output_heights[i]-1)
                    # # heatmaps[i][n, y1:y2, x1:x2, 0] = 1  # Only works for car detection
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
                    heatmaps[i][y_c, x_c, 0] = 1  # Only works for car detection
                    offsets[i][y_c, x_c, 0] = box[k, 0] - x_c * 1.0 / ou_widths[i]
                    offsets[i][y_c, x_c, 1] = box[k, 1] - y_c * 1.0 / ou_heights[i]
                    offsets[i][y_c, x_c, 2] = box[k, 2] - x_c * 1.0 / ou_widths[i]
                    offsets[i][y_c, x_c, 3] = box[k, 3] - y_c * 1.0 / ou_heights[i]

                    x_c = np.int(np.floor(x[k] * ou_widths[i]))
                    y_c = np.int(np.ceil(y[k] * ou_heights[i]))
                    if x_c < 0 or x_c >= self.output_widths[i] or y_c < 0 or y_c >= self.output_heights[i]:
                        continue
                    heatmaps[i][y_c, x_c, 0] = 1  # Only works for car detection
                    offsets[i][y_c, x_c, 0] = box[k, 0] - x_c * 1.0 / ou_widths[i]
                    offsets[i][y_c, x_c, 1] = box[k, 1] - y_c * 1.0 / ou_heights[i]
                    offsets[i][y_c, x_c, 2] = box[k, 2] - x_c * 1.0 / ou_widths[i]
                    offsets[i][y_c, x_c, 3] = box[k, 3] - y_c * 1.0 / ou_heights[i]

                    x_c = np.int(np.ceil(x[k] * ou_widths[i]))
                    y_c = np.int(np.floor(y[k] * ou_heights[i]))
                    if x_c < 0 or x_c >= self.output_widths[i] or y_c < 0 or y_c >= self.output_heights[i]:
                        continue
                    heatmaps[i][y_c, x_c, 0] = 1  # Only works for car detection
                    offsets[i][y_c, x_c, 0] = box[k, 0] - x_c * 1.0 / ou_widths[i]
                    offsets[i][y_c, x_c, 1] = box[k, 1] - y_c * 1.0 / ou_heights[i]
                    offsets[i][y_c, x_c, 2] = box[k, 2] - x_c * 1.0 / ou_widths[i]
                    offsets[i][y_c, x_c, 3] = box[k, 3] - y_c * 1.0 / ou_heights[i]

                    x_c = np.int(np.ceil(x[k] * ou_widths[i]))
                    y_c = np.int(np.ceil(y[k] * ou_heights[i]))
                    if x_c < 0 or x_c >= self.output_widths[i] or y_c < 0 or y_c >= self.output_heights[i]:
                        continue
                    heatmaps[i][y_c, x_c, 0] = 1  # Only works for car detection
                    offsets[i][y_c, x_c, 0] = box[k, 0] - x_c * 1.0 / ou_widths[i]
                    offsets[i][y_c, x_c, 1] = box[k, 1] - y_c * 1.0 / ou_heights[i]
                    offsets[i][y_c, x_c, 2] = box[k, 2] - x_c * 1.0 / ou_widths[i]
                    offsets[i][y_c, x_c, 3] = box[k, 3] - y_c * 1.0 / ou_heights[i]
        else:
            boxes.append([])
        sample = {'images': images, 'orig_image': orig_image, 'depths': depths, 'orig_depth': orig_depth,
                'flows': flows, 'orig_flow': orig_flow, 'heatmaps': heatmaps, 'offsets': offsets}
        if self.transform:
            sample = self.transform(sample)
        return sample

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

    def visualize(self, sample):
        orig_image = sample['orig_image']
        orig_depth = sample['orig_depth']
        orig_flow = sample['orig_flow']
        images = sample['images']
        depths = sample['depths']
        flows = sample['flows']
        heatmaps = sample['heatmaps']
        offsets = sample['offsets']
        # construct a full image containing all scale images
        max_im_size = np.max(np.array(self.im_widths))
        sum_im_size = np.sum(np.array(self.im_heights))
        im_all = np.zeros((sum_im_size, max_im_size, 3))
        dp_all = np.zeros((sum_im_size, max_im_size, 3))
        fl_all = np.zeros((sum_im_size, max_im_size, 3))
        cnt = 0
        for i in range(len(images)):
            im = images[i]
            height, width = im.shape[0], im.shape[1]
            im_all[cnt:cnt + height, 0:width, :] = im
            dp = depths[i][:, :, 0]
            dp = flowlib.visualize_disp(dp)
            dp_all[cnt:cnt + height, 0:width, :] = dp
            fl = flows[i]
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
        hm_all = np.zeros((sum_ou_size, max_ou_size))
        cnt = 0
        for i in range(len(heatmaps)):
            hm = heatmap_maps[i]
            height, width = hm.shape[0], hm.shape[1]
            hm_all[cnt:cnt + height, 0:width] = hm
            cnt = cnt + height
        fig, ax = plt.subplots(1)
        ax.imshow(hm_all)
        plt.show()

        heatmap_on_image = self.visualize_heatmap_on_image(im_all, lb_all)
        fig, ax = plt.subplots(1)
        ax.imshow(heatmap_on_image)
        plt.show()

        b = self.heatmap2box(heatmaps, offsets)
        if len(b) > 0:
            im_height, im_width = orig_image.shape[0], orig_image.shape[1]
            b[:, 0], b[:, 2] = b[:, 0] * im_width, b[:, 2] * im_width
            b[:, 1], b[:, 3] = b[:, 1] * im_height, b[:, 3] * im_height
        fig, ax = plt.subplots(1)
        ax.imshow(orig_image)
        for k in range(b.shape[0]):
            rect = patches.Rectangle((b[k, 0], b[k, 1]), b[k, 2] - 1 - b[k, 0],
                                     b[k, 3] - 1 - b[k, 1], linewidth=2,
                                     edgecolor='g', facecolor='none')
            ax.add_patch(rect)
        plt.show()
        fig, ax = plt.subplots(1)
        orig_depth = flowlib.visualize_disp(orig_depth)
        ax.imshow(orig_depth)
        for k in range(b.shape[0]):
            rect = patches.Rectangle((b[k, 0], b[k, 1]), b[k, 2] - 1 - b[k, 0],
                                     b[k, 3] - 1 - b[k, 1], linewidth=2,
                                     edgecolor='w', facecolor='none')
            ax.add_patch(rect)
        plt.show()
        fig, ax = plt.subplots(1)
        orig_flow = flowlib.visualize_disp(orig_flow)
        orig_flow = flowlib.visualize_flow(orig_flow)
        ax.imshow(orig_flow)
        for k in range(b.shape[0]):
            rect = patches.Rectangle((b[k, 0], b[k, 1]), b[k, 2] - 1 - b[k, 0],
                                     b[k, 3] - 1 - b[k, 1], linewidth=2,
                                     edgecolor='k', facecolor='none')
            ax.add_patch(rect)
        plt.show()


    @staticmethod
    def visualize_heatmap_on_image(image, heatmap):
        im_height, im_width = image.shape[0], image.shape[1]
        heatmap = cv2.resize(heatmap, (im_width, im_height), interpolation=cv2.INTER_LINEAR)
        heatmap = np.dstack((heatmap, heatmap, heatmap))
        heatmap_on_image = image * 0.2 + pred * 0.8
        return heatmap_on_image

    def heatmap2box(self, heatmaps, offsets):
        boxes = []
        for i in range(len(heatmaps)):
            heatmap = heatmaps[i]
            offset = offsets[i]
            ou_height, ou_width = heatmap.shape[0], heatmap.shape[1]
            [y_c, x_c] = np.nonzero(heatmap)
            if len(y_c) > 0:
                for k in range(len(y_c)):
                    of = offset[y_c[k], x_c[k], :]
                    x1 = x_c[k] * 1.0 / ou_width + of[0]
                    y1 = y_c[k] * 1.0 / ou_height + of[1]
                    x2 = x_c[k] * 1.0 / ou_width + of[2]
                    y2 = y_c[k] * 1.0 / ou_height + of[3]
                    boxes.append([x1, y1, x2, y2])
        boxes = np.array(boxes)
        return boxes


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        images = sample['images']
        depths = sample['depths']
        flows = sample['flows']
        heatmaps = sample['heatmaps']
        offsets = sample['offsets']

        images = [torch.from_numpy(im.transpose((2, 0, 1))) for im in images]
        depths = [torch.from_numpy(dp.transpose((2, 0, 1))) for dp in depths]
        flows = [torch.from_numpy(fl.transpose((2, 0, 1))) for fl in flows]
        heatmaps = [torch.from_numpy(hm.transpose((2, 0, 1))) for hm in heatmaps]
        offsets = [torch.from_numpy(of.transpose((2, 0, 1))) for of in offsets]

        sample = {'images': images, 'depths': depths, 'flows': flows,
                  'heatmaps': heatmaps, 'offsets': offsets}
        return sample

data_path = '/mnt/project/yangyi05/kitti/training'
kitti_data = KittiData(data_path)
train_dataset = KittiDataset(kitti_data.train_meta, [128], [384], 1, [16], [48],
                             data_augment=True)
for i in range(2):
    sample = train_dataset[i]
    print(sample['images'][0].shape, sample['depths'][0].shape, sample['flows'][0].shape,
          sample['heatmaps'][0].shape, sample['offsets'][0].shape)

test_dataset = KittiDataset(kitti_data.test_meta, [128], [384], 1, [16], [48])
for i in range(2):
    sample = test_dataset[i]
    print(sample['images'][0].shape, sample['depths'][0].shape, sample['flows'][0].shape,
          sample['heatmaps'][0].shape, sample['offsets'][0].shape)

train_dataset = KittiDataset(kitti_data.train_meta, [128], [384], 1, [16], [48],
                             data_augment=True, transform=transforms.Compose([ToTensor()]))
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
for i, sample in enumerate(train_dataloader):
    if i > 1:
        break
    print(i, sample['images'][0].size(), sample['depths'][0].size(), sample['flows'][0].size(),
          sample['heatmaps'][0].size(), sample['offsets'][0].size())

test_dataset = KittiDataset(kitti_data.test_meta, [128], [384], 1, [16], [48],
                            data_augment=True, transform=transforms.Compose([ToTensor()]))
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)
for i, sample in enumerate(test_dataloader):
    if i > 1:
        break
    print(i, sample['images'][0].size(), sample['depths'][0].size(), sample['flows'][0].size(),
          sample['heatmaps'][0].size(), sample['offsets'][0].size())
