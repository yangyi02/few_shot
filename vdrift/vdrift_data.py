import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import cv2
from scipy.stats.kde import gaussian_kde
import vdrift
import flowlib

import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


class VDriftData(object):
    def __init__(self, data_path, batch_size=8, image_heights=[240], image_widths=[320],
                 output_height=240, output_width=320, num_scale=1,
                 train_proportion=1, test_proportion=1,
                 show_statistics=''):
        self.name = 'vdrift'
        self.image_dir = os.path.join(data_path, 'cam')
        self.depth_dir = os.path.join(data_path, 'depth')
        self.flow_x_dir = os.path.join(data_path, 'flow_x')
        self.flow_y_dir = os.path.join(data_path, 'flow_y')
        self.seg_dir = os.path.join(data_path, 'segcls')

        self.batch_size = batch_size
        self.im_heights = image_heights
        self.im_widths = image_widths
        self.num_scale = num_scale
        self.output_height = output_height
        self.output_width = output_width
        self.orig_im_size = [480, 640]
        self.train_proportion = train_proportion
        self.test_proportion = test_proportion
        self.train_test_split = 0.8

        self.inverse_class_map = {0: 'Road', 1: 'Tree', 2: 'Grass', 3: 'Sky',
                                  4: 'Rock', 5: 'Car', 6: 'Building', 7: 'Mark'}
        self.inverse_color_map = {0: (0, 0, 255), 1: (0, 128, 0), 2: (0, 255, 0), 3: (0, 255, 255),
                                  4: (128, 128, 128), 5: (255, 0, 0), 6: (255, 255, 0), 7: (255, 255, 255)}

        self.meta = self.get_meta()
        self.train_meta, self.test_meta = self.split_train_test()
        logging.info('number of training image: %d, number of testing image: %d',
                     len(self.train_meta['image']), len(self.test_meta['image']))

    def get_meta(self):
        image_files = os.listdir(self.image_dir)
        image_files = sorted(image_files)
        image_file_names = [os.path.join(self.image_dir, f) for f in image_files]
        depth_file_names = [os.path.join(self.depth_dir, f) for f in image_files]
        flow_x_file_names = [os.path.join(self.flow_x_dir, f) for f in image_files]
        flow_y_file_names = [os.path.join(self.flow_y_dir, f) for f in image_files]
        seg_file_names = [os.path.join(self.seg_dir, f) for f in image_files]
        meta = dict()
        meta['image'] = image_file_names
        meta['depth'] = depth_file_names
        meta['flow_x'] = flow_x_file_names
        meta['flow_y'] = flow_y_file_names
        meta['seg'] = seg_file_names
        return meta

    def split_train_test(self):
        num_image, thresh = len(self.meta['image']), self.train_test_split
        train_meta = {'image': self.meta['image'][0:int(num_image * thresh)],
                      'depth': self.meta['depth'][0:int(num_image * thresh)],
                      'flow_x': self.meta['flow_x'][0:int(num_image * thresh)],
                      'flow_y': self.meta['flow_y'][0:int(num_image * thresh)],
                      'seg': self.meta['seg'][0:int(num_image * thresh)]}
        test_meta = {'image': self.meta['image'][int(num_image * thresh):],
                     'depth': self.meta['depth'][int(num_image * thresh):],
                     'flow_x': self.meta['flow_x'][int(num_image * thresh):],
                     'flow_y': self.meta['flow_y'][int(num_image * thresh):],
                     'seg': self.meta['seg'][int(num_image * thresh):]}
        num_train_image = len(train_meta['image'])
        # train_meta['image'] = train_meta['image'][0:int(num_train_image * self.train_proportion)]
        # train_meta['depth'] = train_meta['depth'][0:int(num_train_image * self.train_proportion)]
        # train_meta['flow_x'] = train_meta['flow_x'][0:int(num_train_image * self.train_proportion)]
        # train_meta['flow_y'] = train_meta['flow_y'][0:int(num_train_image * self.train_proportion)]
        # train_meta['seg'] = train_meta['seg'][0:int(num_train_image * self.train_proportion)]
        sample_interval = int(1.0 / self.train_proportion)
        train_meta['image'] = train_meta['image'][0:num_train_image:sample_interval]
        train_meta['depth'] = train_meta['depth'][0:num_train_image:sample_interval]
        train_meta['flow_x'] = train_meta['flow_x'][0:num_train_image:sample_interval]
        train_meta['flow_y'] = train_meta['flow_y'][0:num_train_image:sample_interval]
        train_meta['seg'] = train_meta['seg'][0:num_train_image:sample_interval]
        num_test_image = len(test_meta['image'])
        # test_meta['image'] = test_meta['image'][0:int(num_test_image * self.test_proportion)]
        # test_meta['depth'] = test_meta['depth'][0:int(num_test_image * self.test_proportion)]
        # test_meta['flow_x'] = test_meta['flow_x'][0:int(num_test_image * self.test_proportion)]
        # test_meta['flow_y'] = test_meta['flow_y'][0:int(num_test_image * self.test_proportion)]
        # test_meta['seg'] = test_meta['seg'][0:int(num_test_image * self.test_proportion)]
        sample_interval = int(1.0 / self.test_proportion)
        test_meta['image'] = test_meta['image'][0:num_test_image:sample_interval]
        test_meta['depth'] = test_meta['depth'][0:num_test_image:sample_interval]
        test_meta['flow_x'] = test_meta['flow_x'][0:num_test_image:sample_interval]
        test_meta['flow_y'] = test_meta['flow_y'][0:num_test_image:sample_interval]
        test_meta['seg'] = test_meta['seg'][0:num_test_image:sample_interval]
        return train_meta, test_meta

    def show_basic_statistics(self, status='train'):
        print('Not Implemented')

    def show_full_statistics(self, status='train'):
        print('Not Implemented')

    def data_augmentation(self, image, depth, flow, seg):
        # Flip image
        if np.random.randint(2) == 1:
            image = image[:, ::-1, :]
            depth = depth[:, ::-1]
            flow = flow[:, ::-1, :]
            seg = seg[:, ::-1]
        return image, depth, flow, seg

    def get_next_batch(self, status='train', cnt=0, index=None):
        if status == 'train':
            meta = self.train_meta
        elif status == 'test':
            meta = self.test_meta
        else:
            logging.error('Error: wrong status')
        if index is None:
            index = np.arange(len(meta['image']))
        batch_size, orig_im_size = self.batch_size, self.orig_im_size
        im_heights, im_widths, num_scale = self.im_heights, self.im_widths, self.num_scale
        ou_height, ou_width = self.output_height, self.output_width
        images = [np.zeros((batch_size, im_heights[i], im_widths[i], 3)) for i in range(num_scale)]
        orig_image = np.zeros((batch_size, orig_im_size[0], orig_im_size[1], 3))
        depths = [np.zeros((batch_size, im_heights[i], im_widths[i], 1)) for i in range(num_scale)]
        orig_depth = np.zeros((batch_size, orig_im_size[0], orig_im_size[1], 1))
        flows = [np.zeros((batch_size, im_heights[i], im_widths[i], 2)) for i in range(num_scale)]
        orig_flow = np.zeros((batch_size, orig_im_size[0], orig_im_size[1], 2))
        label = np.zeros((batch_size, ou_height, ou_width))
        orig_label = np.zeros((batch_size, orig_im_size[0], orig_im_size[1]))

        restart = False
        for n in range(batch_size):
            if cnt >= len(index):
                cnt = 0
                restart = True
            image = np.array(Image.open(meta['image'][index[cnt]]))
            image = image / 255.0
            depth = vdrift.decode_depth(meta['depth'][index[cnt]])
            depth = 1.0 / (depth + 0.01)
            # depth = np.log(depth)
            # max_depth = 100
            # depth_image = (depth * 255) / max_depth
            # depth_image = np.clip(depth_image, 0, 255)
            # depth = depth_image / 255.0
            flow = vdrift.decode_flow(meta['flow_x'][index[cnt]], meta['flow_y'][index[cnt]])
            seg = np.array(Image.open(meta['seg'][index[cnt]]))

            if status == 'train':
                image, depth, flow, seg = self.data_augmentation(image, depth, flow, seg)
            im_height, im_width = image.shape[0], image.shape[1]

            for i in range(num_scale):
                images[i][n, :, :, :] = cv2.resize(image, (im_widths[i], im_heights[i]),
                                                   interpolation=cv2.INTER_AREA)
                depths[i][n, :, :, 0] = cv2.resize(depth, (im_widths[i], im_heights[i]),
                                                   interpolation=cv2.INTER_AREA)
                flows[i][n, :, :, :] = cv2.resize(flow, (im_widths[i], im_heights[i]),
                                                  interpolation=cv2.INTER_AREA)
            label[n, :, :] = cv2.resize(seg, (ou_width, ou_height),
                                        interpolation=cv2.INTER_NEAREST)
            orig_image[n, :, :, :] = image
            orig_depth[n, :, :, 0] = depth
            orig_flow[n, :, :, :] = flow
            orig_label[n, :, :] = seg
            cnt = cnt + 1
        for i in range(num_scale):
            images[i] = images[i].transpose((0, 3, 1, 2))
            depths[i] = depths[i].transpose((0, 3, 1, 2))
            flows[i] = flows[i].transpose((0, 3, 1, 2))
        orig_image = orig_image.transpose((0, 3, 1, 2))
        orig_depth = orig_depth.transpose((0, 3, 1, 2))
        orig_flow = orig_flow.transpose((0, 3, 1, 2))
        return images, orig_image, depths, orig_depth, flows, orig_flow, label, orig_label, \
            cnt, restart

    def get_one_sample(self, image_name, depth_name, flow_x_name, flow_y_name, seg_name):
        image = np.array(Image.open(image_name))
        image = image / 255.0
        depth = vdrift.decode_depth(depth_name)
        depth = 1.0 / (depth + 0.01)
        # depth = np.log(depth)
        # max_depth = 100
        # depth_image = (depth * 255) / max_depth
        # depth_image = np.clip(depth_image, 0, 255)
        # depth = depth_image / 255.0
        flow = vdrift.decode_flow(flow_x_name, flow_y_name)
        seg = np.array(Image.open(seg_name))

        batch_size = 1
        orig_im_size = self.orig_im_size
        im_heights, im_widths, num_scale = self.im_heights, self.im_widths, self.num_scale
        ou_height, ou_width = self.output_height, self.output_width
        images = [np.zeros((batch_size, im_heights[i], im_widths[i], 3)) for i in range(num_scale)]
        orig_image = np.zeros((batch_size, orig_im_size[0], orig_im_size[1], 3))
        depths = [np.zeros((batch_size, im_heights[i], im_widths[i], 1)) for i in range(num_scale)]
        orig_depth = np.zeros((batch_size, orig_im_size[0], orig_im_size[1], 1))
        flows = [np.zeros((batch_size, im_heights[i], im_widths[i], 2)) for i in range(num_scale)]
        orig_flow = np.zeros((batch_size, orig_im_size[0], orig_im_size[1], 2))
        label = np.zeros((batch_size, ou_height, ou_width))
        orig_label = np.zeros((batch_size, orig_im_size[0], orig_im_size[1]))

        im_height, im_width = image.shape[0], image.shape[1]

        for i in range(num_scale):
            images[i][0, :, :, :] = cv2.resize(image, (im_widths[i], im_heights[i]),
                                               interpolation=cv2.INTER_AREA)
            depths[i][0, :, :, 0] = cv2.resize(depth, (im_widths[i], im_heights[i]),
                                               interpolation=cv2.INTER_AREA)
            flows[i][0, :, :, :] = cv2.resize(flow, (im_widths[i], im_heights[i]),
                                              interpolation=cv2.INTER_AREA)
        label[0, :, :] = cv2.resize(seg, (ou_width, ou_height),
                                    interpolation=cv2.INTER_NEAREST)
        orig_image[0, :, :, :] = image
        orig_depth[0, :, :, 0] = depth
        orig_flow[0, :, :, :] = flow
        orig_label[0, :, :] = seg
        for i in range(num_scale):
            images[i] = images[i].transpose((0, 3, 1, 2))
            depths[i] = depths[i].transpose((0, 3, 1, 2))
            flows[i] = flows[i].transpose((0, 3, 1, 2))
        orig_image = orig_image.transpose((0, 3, 1, 2))
        orig_depth = orig_depth.transpose((0, 3, 1, 2))
        orig_flow = orig_flow.transpose((0, 3, 1, 2))
        return images, orig_image, depths, orig_depth, flows, orig_flow, label, orig_label

    def visualize(self, images, orig_image, depths, orig_depth, flows, orig_flow, label, orig_label, idx=None):
        if idx is None:
            idx = 0
        orig_im = orig_image[idx, :, :, :].transpose(1, 2, 0)
        orig_dp = orig_depth[idx, 0, :, :]
        orig_fl = orig_flow[idx, :, :, :].transpose(1, 2, 0)
        orig_lb = orig_label[idx, :, :]

        fig, ax = plt.subplots(1)
        ax.imshow(orig_im)
        plt.show()
        fig, ax = plt.subplots(1)
        orig_dp = flowlib.visualize_disp(orig_dp)
        ax.imshow(orig_dp)
        plt.show()
        fig, ax = plt.subplots(1)
        orig_fl = flowlib.visualize_flow(orig_fl)
        ax.imshow(orig_fl)
        plt.show()
        fig, ax = plt.subplots(1)
        orig_lb = vdrift.visualize_seg(orig_lb, self.inverse_color_map)
        ax.imshow(orig_lb)
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

        seg = vdrift.visualize_seg(label[idx, :, :], self.inverse_color_map)
        fig, ax = plt.subplots(1)
        ax.imshow(seg.astype(np.uint8))
        plt.show()

        seg_on_image = self.visualize_seg_on_image(im_all, seg)
        fig, ax = plt.subplots(1)
        ax.imshow(seg_on_image)
        plt.show()

    def visualize_seg_on_image(self, image, seg):
        im_height, im_width = image.shape[0], image.shape[1]
        seg = cv2.resize(seg, (im_width, im_height), interpolation=cv2.INTER_NEAREST)
        if np.max(image) > 1.01:
            image = image / 255.0
        if np.max(seg) > 1.01:
            seg = seg / 255.0
        # seg = vdrift.visualize_seg(seg, self.inverse_color_map)
        seg_on_image = image * 0.5 + seg * 0.5
        return seg_on_image
