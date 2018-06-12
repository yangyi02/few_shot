import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import cv2
from scipy.stats.kde import gaussian_kde

import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


class MLTData(object):
    def __init__(self, batch_size=32, image_size=(256, 128), direction_type='gt',
                 train_proportion=1, test_proportion=1, show_statistics=''):
        self.name = 'mlt'
        self.img_dir = '/media/yi/DATA/data-orig/MLT/image'
        self.depth_dir = '/media/yi/DATA/data-orig/MLT/depth'
        self.box_dir = '/home/yi/code/few_shot/mlt/box'

        self.batch_size = batch_size
        self.im_size = image_size
        self.orig_im_size = [480, 640]
        self.direct_type = direction_type
        self.train_proportion = train_proportion
        self.test_proportion = test_proportion
        self.train_test_split = 0.7

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

        self.class_map = {3: 0, 4: 1, 5: 2, 6: 3, 13: 4, 24: 5}
        self.inverse_class_map = {0: 'Bed', 1: 'Chair', 2: 'Sofa', 3: 'Table', 4: 'Desks',
                                  5: 'Television'}

    def get_meta(self):
        meta = {'img': [], 'depth': [], 'box': []}
        for sub_dir in os.listdir(self.box_dir):
            box_files = os.listdir(os.path.join(self.box_dir, sub_dir))
            box_files.sort(key=lambda f: int(filter(str.isdigit, f)))
            box_file_names = [os.path.join(self.box_dir, sub_dir, f) for f in box_files]
            img_files, depth_files = [], []
            for f in box_files:
                file_name, file_ext = os.path.splitext(f)
                img_files.append(file_name + '_color.jpg')
                depth_files.append(file_name + '_depth.png')
            img_file_names = [os.path.join(self.img_dir, sub_dir, f) for f in img_files]
            depth_file_names = [os.path.join(self.depth_dir, sub_dir, f) for f in depth_files]
            meta['img'].extend(img_file_names)
            meta['depth'].extend(depth_file_names)
            meta['box'].extend(box_file_names)
        return meta

    def split_train_test(self):
        num_image, thresh = len(self.meta['img']), self.train_test_split
        train_meta = {'img': self.meta['img'][0:int(num_image * thresh)],
                      'depth': self.meta['depth'][0:int(num_image * thresh)],
                      'box': self.meta['box'][0:int(num_image * thresh)]}
        test_meta = {'img': self.meta['img'][int(num_image * thresh):],
                     'depth': self.meta['depth'][int(num_image * thresh):],
                     'box': self.meta['box'][int(num_image * thresh):]}
        num_train_image = len(train_meta['img'])
        train_meta['img'] = train_meta['img'][0:int(num_image * self.train_proportion)]
        train_meta['depth'] = train_meta['depth'][0:int(num_image * self.train_proportion)]
        train_meta['box'] = train_meta['box'][0:int(num_image * self.train_proportion)]
        num_test_image = len(test_meta['img'])
        test_meta['img'] = test_meta['img'][0:int(num_image * self.test_proportion)]
        test_meta['depth'] = test_meta['depth'][0:int(num_image * self.test_proportion)]
        test_meta['box'] = test_meta['box'][0:int(num_image * self.test_proportion)]
        return train_meta, test_meta

    def rearrange_annotation(self, meta):
        anno = {'img': [], 'depth': [], 'box': [], 'label': []}
        for i in range(len(meta['box'])):
            box_and_label = []
            with open(meta['box'][i]) as txt_file:
                box_info = txt_file.readlines()
            for row in box_info:
                row = row.strip().split(' ')
                box_and_label.append(row)
            box_and_label = np.array(box_and_label).astype(np.int)
            for n in range(box_and_label.shape[0]):
                anno['img'].append(meta['img'][i])
                anno['depth'].append(meta['depth'][i])
                anno['box'].append(box_and_label[n, 0:4])
                anno['label'].append(box_and_label[n, 4])
        return anno

    def show_basic_statistics(self, status='train'):
        if status == 'train':
            anno = self.train_anno
        elif status == 'test':
            anno = self.test_anno
        else:
            logging.error('Error: wrong status')
        labels = set(list(anno['label']))
        count = dict()
        max_count = 0
        total_count = 0
        for label in labels:
            count[label] = (anno['label'] == label).sum()
            if count[label] > max_count:
                max_count = count[label]
            total_count = total_count + count[label]
        print(status, count, max_count * 1.0 / total_count)

    def show_full_statistics(self, status='train'):
        self.show_basic_statistics(status)
        if status == 'train':
            anno = self.train_anno
        elif status == 'test':
            anno = self.test_anno
        else:
            logging.error('Error: wrong status')
        box = np.array(anno['box'])
        x = (box[:, 0] + box[:, 2]) / 2.0 / self.orig_im_size[1]
        y = (box[:, 1] + box[:, 3]) / 2.0 / self.orig_im_size[0]
        w = (box[:, 2] - box[:, 0]) * 1.0 / self.orig_im_size[1]
        h = (box[:, 3] - box[:, 1]) * 1.0 / self.orig_im_size[0]
        print(np.min(x), np.max(x), np.mean(x), np.median(x))
        print(np.min(y), np.max(y), np.mean(y), np.median(y))
        print(np.min(w), np.max(w), np.mean(w), np.median(w))
        print(np.min(h), np.max(h), np.mean(h), np.median(h))
        fig, ax = plt.subplots(1)
        ax.scatter(x, y)
        fig, ax = plt.subplots(1)
        ax.scatter(w, h)

        num_bins = 100
        fig, ax = plt.subplots(1)
        counts, bin_edges = np.histogram(x, bins=num_bins, range=[0, 1], normed=True)
        cdf = np.cumsum(counts)
        ax.plot(bin_edges[1:], cdf)
        counts, bin_edges = np.histogram(y, bins=num_bins, range=[0, 1], normed=True)
        cdf = np.cumsum(counts)
        fig, ax = plt.subplots(1)
        ax.plot(bin_edges[1:], cdf)
        counts, bin_edges = np.histogram(w, bins=num_bins, range=[0, 1], normed=True)
        cdf = np.cumsum(counts)
        fig, ax = plt.subplots(1)
        ax.plot(bin_edges[1:], cdf)
        counts, bin_edges = np.histogram(h, bins=num_bins, range=[0, 1], normed=True)
        cdf = np.cumsum(counts)
        fig, ax = plt.subplots(1)
        ax.plot(bin_edges[1:], cdf)

        fig, ax = plt.subplots(1)
        k = gaussian_kde(np.vstack([x, y]))
        xi, yi = np.mgrid[0:1:0.01, 0:1:0.01]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        # ax.pcolormesh(xi, yi, zi.reshape(xi.shape))
        ax.contourf(xi, yi, zi.reshape(xi.shape))
        fig, ax = plt.subplots(1)
        k = gaussian_kde(np.vstack([w, h]))
        xi, yi = np.mgrid[0:1:0.01, 0:1:0.01]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        # ax.pcolormesh(xi, yi, zi.reshape(xi.shape))
        ax.contourf(xi, yi, zi.reshape(xi.shape))
        plt.show()

    def data_augmentation(self, image, depth, box):
        im_width = image.shape[1]
        # Flip image
        if np.random.randint(2) == 1:
            image = image[:, ::-1, :]
            depth = depth[:, ::-1]
            flip_box = np.zeros(4)
            flip_box[0] = im_width - box[2]
            flip_box[1] = box[1]
            flip_box[2] = im_width - box[0]
            flip_box[3] = box[3]
            box = flip_box
        return image, depth, box

    def get_next_batch(self, status='train', cnt=0, index=None):
        if status == 'train':
            anno = self.train_anno
        elif status == 'test':
            anno = self.test_anno
        else:
            logging.error('Error: wrong status')
        if index is None:
            index = np.arange(len(anno['img']))
        batch_size, im_size, orig_im_size = self.batch_size, self.im_size, self.orig_im_size
        images = [np.zeros((batch_size, i_s, i_s, 3)) for i_s in im_size]
        orig_images = np.zeros((batch_size, orig_im_size[0], orig_im_size[1], 3))
        depths = [np.zeros((batch_size, i_s, i_s, 1)) for i_s in im_size]
        orig_depths = np.zeros((batch_size, orig_im_size[0], orig_im_size[1], 1))
        directions = np.zeros((batch_size, 4))
        boxes = np.zeros((batch_size, 4))
        labels = np.zeros(batch_size)
        restart = False
        for n in range(batch_size):
            if cnt >= len(index):
                cnt = 0
                restart = True
            image = np.array(Image.open(anno['img'][index[cnt]]))
            im_height, im_width = image.shape[0], image.shape[1]
            image = image / 255.0
            depth = np.array(Image.open(anno['depth'][index[cnt]]))
            depth = depth / 5000.0
            box = np.array(anno['box'][index[cnt]])
            label = anno['label'][index[cnt]]

            if status == 'train':
                image, depth, box = self.data_augmentation(image, depth, box)

            for i in range(len(im_size)):
                images[i][n, :, :, :] = cv2.resize(image, (im_size[i], im_size[i]),
                                                   interpolation=cv2.INTER_AREA)
                depths[i][n, :, :, 0] = cv2.resize(depth, (im_size[i], im_size[i]),
                                                   interpolation=cv2.INTER_AREA)
            orig_images[n, :, :, :] = image
            orig_depths[n, :, :, 0] = depth
            boxes[n, 0] = box[0] * 1.0 / im_width
            boxes[n, 1] = box[1] * 1.0 / im_height
            boxes[n, 2] = box[2] * 1.0 / im_width
            boxes[n, 3] = box[3] * 1.0 / im_height
            x = (box[0] + box[2]) * 1.0 / 2 / im_width
            y = (box[1] + box[3]) * 1.0 / 2 / im_height
            w = (box[2] - box[0]) * 1.0 / im_width
            h = (box[3] - box[1]) * 1.0 / im_height
            if self.direct_type == 'gt':
                directions[n, :] = [x, y, w, h]
            else:
                num = int(self.direct_type)
                directions[n, :] = [int(x * num), int(y * num), int(w * num), int(h * num)]
            labels[n] = self.class_map[label]
            cnt = cnt + 1
        for i in range(len(im_size)):
            images[i] = images[i].transpose((0, 3, 1, 2))
            depths[i] = depths[i].transpose((0, 3, 1, 2))
        orig_images = orig_images.transpose((0, 3, 1, 2))
        orig_depths = orig_depths.transpose((0, 3, 1, 2))
        return images, orig_images, depths, orig_depths, boxes, directions, labels, cnt, restart

    def get_one_sample(self, image_name, depth_name, box_name):
        image = np.array(Image.open(image_name))
        im_height, im_width = image.shape[0], image.shape[1]
        image = image / 255.0
        depth = np.array(Image.open(depth_name))
        depth = depth / 5000.0

        box_and_label = []
        with open(box_name) as txt_file:
            box_info = txt_file.readlines()
        for row in box_info:
            row = row.strip().split(' ')
            box_and_label.append(row)
        box_and_label = np.array(box_and_label).astype(np.int)
        box, label = box_and_label[:, 0:4], box_and_label[:, 4]

        batch_size = box.shape[0]
        im_size, orig_im_size = self.im_size, self.orig_im_size
        images = [np.zeros((batch_size, i_s, i_s, 3)) for i_s in im_size]
        orig_images = np.zeros((batch_size, orig_im_size[0], orig_im_size[1], 3))
        depths = [np.zeros((batch_size, i_s, i_s, 1)) for i_s in im_size]
        orig_depths = np.zeros((batch_size, orig_im_size[0], orig_im_size[1], 1))
        directions = np.zeros((batch_size, 4))
        boxes = np.zeros((batch_size, 4))
        labels = np.zeros(batch_size)

        for n in range(batch_size):
            for i in range(len(im_size)):
                images[i][n, :, :, :] = cv2.resize(image, (im_size[i], im_size[i]),
                                                   interpolation=cv2.INTER_AREA)
                depths[i][n, :, :, 0] = cv2.resize(depth, (im_size[i], im_size[i]),
                                                   interpolation=cv2.INTER_AREA)
            orig_images[n, :, :, :] = image
            orig_depths[n, :, :, 0] = depth
            boxes[n, 0] = box[n, 0] * 1.0 / im_width
            boxes[n, 1] = box[n, 1] * 1.0 / im_height
            boxes[n, 2] = box[n, 2] * 1.0 / im_width
            boxes[n, 3] = box[n, 3] * 1.0 / im_height
            x = (box[n, 0] + box[n, 2]) * 1.0 / 2 / im_width
            y = (box[n, 1] + box[n, 3]) * 1.0 / 2 / im_height
            w = (box[n, 2] - box[n, 0]) * 1.0 / im_width
            h = (box[n, 3] - box[n, 1]) * 1.0 / im_height
            if self.direct_type == 'gt':
                directions[n, :] = [x, y, w, h]
            else:
                num = int(self.direct_type)
                directions[n, :] = [int(x * num), int(y * num), int(w * num), int(h * num)]
            labels[n] = self.class_map[label[n]]
        for i in range(len(im_size)):
            images[i] = images[i].transpose((0, 3, 1, 2))
            depths[i] = depths[i].transpose((0, 3, 1, 2))
        orig_images = orig_images.transpose((0, 3, 1, 2))
        orig_depths = orig_depths.transpose((0, 3, 1, 2))
        return images, orig_images, depths, orig_depths, boxes, directions, labels

    def visualize(self, images, orig_images, depths, orig_depths, boxes, directions, labels, idx=None):
        if idx is None:
            idx = range(len(labels))
        else:
            idx = [idx]
        for n in idx:
            print(directions[n, :], labels[n], self.inverse_class_map[labels[n]])
            # Plot original image and depth with bounding box
            orig_im = orig_images[n, :, :, :].transpose(1, 2, 0)
            orig_dp = orig_depths[n, 0, :, :]
            b = boxes[n, :].copy()
            im_height, im_width = orig_im.shape[0], orig_im.shape[1]
            b[0], b[2] = b[0] * im_width, b[2] * im_width
            b[1], b[3] = b[1] * im_height, b[3] * im_height
            fig, ax = plt.subplots(1)
            ax.imshow(orig_im)
            rect = patches.Rectangle((b[0], b[1]), b[2] - 1 - b[0], b[3] - 1 - b[1], linewidth=2,
                                     edgecolor='g', facecolor='none')
            ax.add_patch(rect)
            fig, ax = plt.subplots(1)
            ax.imshow(orig_dp)
            rect = patches.Rectangle((b[0], b[1]), b[2] - 1 - b[0], b[3] - 1 - b[1], linewidth=2,
                                     edgecolor='w', facecolor='none')
            ax.add_patch(rect)
            # construct a full image containing all scale images
            max_im_size = np.max(np.array(self.im_size))
            sum_im_size = np.sum(np.array(self.im_size))
            im_all = np.zeros((max_im_size, sum_im_size, 3))
            dp_all = np.zeros((max_im_size, sum_im_size))
            cnt = 0
            for i in range(len(images)):
                im = images[i][n, :, :, :].transpose(1, 2, 0)
                height, width = im.shape[0], im.shape[1]
                im_all[0:height, cnt:cnt + width, :] = im
                dp = depths[i][0, :, :]
                dp_all[0:height, cnt:cnt + width] = dp
                cnt = cnt + width
            fig, ax = plt.subplots(1)
            ax.imshow(im_all)
            fig, ax = plt.subplots(1)
            ax.imshow(dp_all)
            plt.show()
