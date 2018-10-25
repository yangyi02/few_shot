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


class CityScapeData(object):
    def __init__(self, file_list, batch_size=8, image_heights=[256], image_widths=[512],
                 output_height=256, output_width=512, num_scale=1, proportion=1,
                 data_augment=False, shuffle=False):
        self.name = 'cityscape'

        self.batch_size = batch_size
        self.im_heights = image_heights
        self.im_widths = image_widths
        self.num_scale = num_scale
        self.output_height = output_height
        self.output_width = output_width
        self.orig_im_size = [1024, 2048]
        self.proportion = proportion
        self.data_augment = data_augment
        self.meta = self.get_meta(file_list)
        self.num_data = len(self.meta['image'])
        self.num_batch = int(np.ceil(self.num_data * 1.0 / batch_size))
        self.last_batch_size = self.num_data - (self.num_batch - 1) * batch_size
        if shuffle:
            self.sample_index = np.random.permutation(self.num_data)
        else:
            self.sample_index = np.arange(0, self.num_data)

        self.inverse_class_map = {0: 'Road', 1: 'Tree', 2: 'Grass', 3: 'Sky',
                                  4: 'Rock', 5: 'Car', 6: 'Building', 7: 'Mark'}
        self.inverse_color_map = {0: (0, 0, 255), 1: (0, 128, 0), 2: (0, 255, 0), 3: (0, 255, 255),
                                  4: (128, 128, 128), 5: (255, 0, 0), 6: (255, 255, 0), 7: (255, 255, 255)}

    def get_meta(self, file_list):
        lines = open(file_list, 'r').readlines()
        meta = {'image': [], 'depth': [], 'flow': [], 'seg': []}
        for line in lines:
            line = line.strip().split(' ')
            meta['image'].append(line[0])
            meta['depth'].append(line[1])
            meta['seg'].append(line[2])
        num_image = len(meta['image'])
        sample_interval = int(1.0 / self.proportion)
        meta['image'] = meta['image'][0:num_image:sample_interval]
        meta['depth'] = meta['depth'][0:num_image:sample_interval]
        meta['seg'] = meta['seg'][0:num_image:sample_interval]
        return meta

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
        ou_height, ou_width = self.output_height, self.output_width
        images = [np.zeros((batch_size, im_heights[i], im_widths[i], 3)) for i in range(num_scale)]
        depths = [np.zeros((batch_size, im_heights[i], im_widths[i], 1)) for i in range(num_scale)]
        flows = [np.zeros((batch_size, im_heights[i], im_widths[i], 2)) for i in range(num_scale)]
        label = np.zeros((batch_size, ou_height, ou_width))

        for n in range(batch_size):
            data_idx = self.sample_index[batch_idx * self.batch_size + n]
            image = np.array(Image.open(self.meta['image'][data_idx]))
            image = image / 255.0
            depth = flowlib.read_disp_png(self.meta['depth'][data_idx])
            seg = np.array(Image.open(self.meta['seg'][data_idx]))

            if self.data_augment:
                image, depth, flow, seg = self.data_augmentation(image, depth, flow, seg)

            for i in range(num_scale):
                images[i][n, :, :, :] = cv2.resize(image, (im_widths[i], im_heights[i]),
                                                   interpolation=cv2.INTER_AREA)
                depths[i][n, :, :, 0] = cv2.resize(depth, (im_widths[i], im_heights[i]),
                                                   interpolation=cv2.INTER_AREA)
                # flows[i][n, :, :, :] = cv2.resize(flow, (im_widths[i], im_heights[i]),
                #                                   interpolation=cv2.INTER_AREA)
            label[n, :, :] = cv2.resize(seg, (ou_width, ou_height),
                                        interpolation=cv2.INTER_NEAREST)
        sample = {'images': images, 'depths': depths, 'flows': flows, 'seg': label}
        return sample

    @staticmethod
    def data_augmentation(image, depth, flow, seg):
        # Flip image
        if np.random.randint(2) == 1:
            image = image[:, ::-1, :]
            depth = depth[:, ::-1]
            flow = flow[:, ::-1, :]
            seg = seg[:, ::-1]
        return image, depth, flow, seg

    def get_one_sample(self, image_name, depth_name, flow_name, seg_name):
        batch_size, orig_im_size = 1, self.orig_im_size

        im_heights, im_widths, num_scale = self.im_heights, self.im_widths, self.num_scale
        ou_height, ou_width = self.output_height, self.output_width
        images = [np.zeros((batch_size, im_heights[i], im_widths[i], 3)) for i in range(num_scale)]
        depths = [np.zeros((batch_size, im_heights[i], im_widths[i], 1)) for i in range(num_scale)]
        flows = [np.zeros((batch_size, im_heights[i], im_widths[i], 3)) for i in range(num_scale)]
        label = np.zeros((batch_size, ou_height, ou_width))

        for n in range(batch_size):
            image = np.array(Image.open(image_name))
            image = image / 255.0
            depth = flowlib.read_disp_png(depth_name)
            seg = np.array(Image.open(seg_name))

            for i in range(num_scale):
                images[i][0, :, :, :] = cv2.resize(image, (im_widths[i], im_heights[i]),
                                                   interpolation=cv2.INTER_AREA)
                depths[i][0, :, :, 0] = cv2.resize(depth, (im_widths[i], im_heights[i]),
                                                   interpolation=cv2.INTER_AREA)
                # flows[i][0, :, :, :] = cv2.resize(flow, (im_widths[i], im_heights[i]),
                #                                   interpolation=cv2.INTER_AREA)
            label[0, :, :] = cv2.resize(seg, (ou_width, ou_height),
                                        interpolation=cv2.INTER_NEAREST)
        sample = {'images': images, 'flows': flows, 'depths': depths, 'seg': label}
        return sample
