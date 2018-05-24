import os
import math
import numpy
import matplotlib.pyplot as plt
from PIL import Image
from skimage import io, transform
import pickle
import cv2

import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                            level=logging.INFO)


class RealData(object):
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.im_size = args.image_size
        self.im_channel = args.image_channel
        self.m_range = args.motion_range
        self.num_frame = args.num_frame
        self.m_dict, self.reverse_m_dict, self.m_kernel = self.motion_dict()
        self.visualizer = BaseVisualizer(args, self.reverse_m_dict)
        self.save_display = args.save_display
        self.save_display_dir = args.save_display_dir
        self.min_diff_thresh = args.min_diff_thresh
        self.max_diff_thresh = args.max_diff_thresh
        self.diff_div_thresh = args.diff_div_thresh
        self.fixed_data = args.fixed_data
        if args.fixed_data:
            numpy.random.seed(args.seed)
        self.rand_noise = args.rand_noise
        self.augment_reverse = args.augment_reverse

    def get_meta(self, image_dir):
        meta = {}
        meta['box_dir'] = os.path.join(image_dir, 'bb')
        meta['cls_dir'] = os.path.join(image_dir, 'cls')
        meta['img_dir'] = os.path.join(image_dir, 'img/1')
        meta['inst_dir'] = os.path.join(image_dir, 'inst')
        meta['img'], meta['box'], meta['cls'], meta['inst'] = [], [], [], []

        for sub_dir in os.listdir(meta['img_dir']):
            img_files = os.listdir(os.path.join(meta['img_dir'], sub_dir))
            img_files.sort(key=lambda f: int(filter(str.isdigit, f)))
            img_file_names = [os.path.join(meta['img_dir'], sub_dir, f) for f in img_files]
            box_files = os.listdir(os.path.join(meta['box_dir'], sub_dir))
            box_files.sort(key=lambda f: int(filter(str.isdigit, f)))
            box_file_names = [os.path.join(meta['box_dir'], sub_dir, f) for f in box_files]
            cls_files = os.listdir(os.path.join(meta['cls_dir'], sub_dir))
            cls_files.sort(key=lambda f: int(filter(str.isdigit, f)))
            cls_file_names = [os.path.join(meta['cls_dir'], sub_dir, f) for f in cls_files]
            inst_files = os.listdir(os.path.join(meta['inst_dir'], sub_dir))
            inst_files.sort(key=lambda f: int(filter(str.isdigit, f)))
            inst_file_names = [os.path.join(meta['inst_dir'], sub_dir, f) for f in inst_files]
            meta['img'].extend(img_files)
            meta['box'].extend(box_files)
            meta['cls'].extend(cls_files)
            meta['inst'].exten(inst_files)
        return meta

    def generate_data(self, meta):
        batch_size, im_size = self.batch_size, self.im_size
        idx = numpy.random.permutation(len(meta['img']))
        im = numpy.zeros((batch_size, 3, im_size, im_size))
        i, cnt = 0, 0
        while i < batch_size:
            image_name = meta['img'][idx[cnt]]
            image = numpy.array(Image.open(image_name))
            image = image / 255.0
            height, width = image.shape[0], image.shape[1]
            box = []
            with open(meta['box'][idx[cnt]]) as csv_file:
                box_info = csv.reader(csv_file, delimiter=',')
                for row in box_info:
                    box.append(row)

            idx_h = numpy.random.randint(0, height + 1 - im_size)
            idx_w = numpy.random.randint(0, width + 1 - im_size)
            image = image.transpose((2, 0, 1))
            im[i, j, :, :, :] = image[:, idx_h:idx_h+im_size, idx_w:idx_w+im_size]
        return im

    def display(self, im):
        num_frame, im_channel = self.num_frame, self.im_channel
        width, height = self.visualizer.get_img_size(2, num_frame)
        img = numpy.ones((height, width, 3))
        prev_im = None
        for i in range(num_frame):
            curr_im = im[0, i, :, :, :].transpose(1, 2, 0)
            x1, y1, x2, y2 = self.visualizer.get_img_coordinate(1, i + 1)
            img[y1:y2, x1:x2, :] = curr_im

            if i > 0:
                im_diff = abs(curr_im - prev_im)
                x1, y1, x2, y2 = self.visualizer.get_img_coordinate(2, i + 1)
                img[y1:y2, x1:x2, :] = im_diff
            prev_im = curr_im

        if self.save_display:
            img = img * 255.0
            img = img.astype(numpy.uint8)
            img = Image.fromarray(img)
            img.save(os.path.join(self.save_display_dir, 'data.png'))
        else:
            plt.figure(1)
            plt.imshow(img)
            plt.axis('off')
            plt.show()
