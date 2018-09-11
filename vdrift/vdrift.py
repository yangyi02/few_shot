import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import flowlib


def decode_depth(filename):
    im = np.array(Image.open(filename))
    depth = 2000.0 * (256 * 256 * im[:, :, 2] + 256 * im[:, :, 1] + im[:, :, 0]) / (256**3)
    return depth


def decode_flow(filename_x, filename_y):
    im_x = np.array(Image.open(filename_x))
    im_y = np.array(Image.open(filename_y))
    height, width = im_x.shape[0], im_x.shape[1]
    flow = np.zeros((height, width, 2))
    vx = 256 * 256 * im_x[:, :, 2] + 256 * im_x[:, :, 1] + im_x[:, :, 0]
    vy = 256 * 256 * im_y[:, :, 2] + 256 * im_y[:, :, 1] + im_y[:, :, 0]
    flow[:, :, 0] = vx / 10000.0 - 300.0
    flow[:, :, 1] = vy / 10000.0 - 300.0
    return flow


def visualize_seg(seg, inverse_color_map):
    height, width = seg.shape[0], seg.shape[1]
    seg_new_0 = np.zeros_like(seg)
    seg_new_1 = np.zeros_like(seg)
    seg_new_2 = np.zeros_like(seg)
    for key in inverse_color_map.keys():
        mask = seg == key
        seg_new_0[mask] = inverse_color_map[key][0]
        seg_new_1[mask] = inverse_color_map[key][1]
        seg_new_2[mask] = inverse_color_map[key][2]
    seg_new = np.dstack((seg_new_0, seg_new_1, seg_new_2))
    seg_new = seg_new.astype(np.uint8)
    return seg_new

def main():
    data_dir = '/media/yi/DATA/data-orig/vdrift/scripts/images'
    im = np.array(Image.open(os.path.join(data_dir, 'cam.png')))
    plt.subplots()
    plt.imshow(im)
    plt.show()

    depth = decode_depth(os.path.join(data_dir, 'depth.png'))
    max_depth = 100
    depth_image = (depth * 255) / max_depth
    depth_image = np.clip(depth_image, 0, 255)
    depth_image = depth_image.astype(np.uint8)
    plt.subplots()
    plt.imshow(depth_image)
    plt.show()

    flow = decode_flow(os.path.join(data_dir, 'flow_x.png'), os.path.join(data_dir, 'flow_y.png'))
    flow_image= flowlib.visualize_flow(flow, 10)
    plt.subplots()
    plt.imshow(flow_image)
    plt.show()

    seg = np.array(Image.open(os.path.join(data_dir, 'seg.png')))
    plt.subplots()
    plt.imshow(seg)
    plt.show()


if __name__ == '__main__':
    main()
