import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import flowlib
from nms import heatmap2box


def visualize_input(sample, idx=None, display=False):
    if idx is None:
        idx = 0
    # Visualize input images in all scales
    images = [im[idx, :, :, :] for im in sample['images']]
    image_all = concat_images(images)
    if display:
        visualize_image(image_all)
    # Visualize input depths in all scales
    depths = [flowlib.visualize_disp(dp[idx, :, :, 0]) for dp in sample['depths']]
    depth_all = concat_images(depths)
    if display:
        visualize_image(depth_all)
    # Visualize input flows in all scales
    flows = [flowlib.visualize_flow(fl[idx, :, :, :]) for fl in sample['flows']]
    flow_all = concat_images(flows)
    if display:
        visualize_image(flow_all.astype(np.uint8))


def visualize_heatmap(sample, heatmaps=None, idx=None, display=False):
    # Visualize output heatmaps in all scales
    if idx is None:
        idx = 0
    # Visualize input images in all scales
    images = [im[idx, :, :, :] for im in sample['images']]
    image_all = concat_images(images)
    if heatmaps is None:
        heatmaps = [hm[idx, :, :, 0] for hm in sample['heatmaps']]
    else:
        heatmaps = [hm[idx, :, :, 0] for hm in heatmaps]
    heatmap_all = concat_images(heatmaps)
    if display:
        visualize_image(heatmap_all)
    heatmap_on_image = concat_heatmap_on_image(image_all, heatmap_all)
    if display:
        visualize_image(heatmap_on_image)
    return heatmap_on_image


def visualize_box(sample, boxes=None, idx=None, display=False):
    if idx is None:
        idx = 0
    if boxes is None:
        heatmaps = [hm[idx, :, :, 0] for hm in sample['heatmaps']]
        offsets = [of[idx, :, :, :] for of in sample['offsets']]
        boxes = heatmap2box(heatmaps, offsets)
    else:
        boxes = boxes[idx]
    # Visualize reconstructed bounding boxes on image in original scale
    orig_image = sample['orig_image'][idx, :, :, :]
    box_on_image = concat_box_on_image(orig_image, boxes, color=(0, 255, 0))
    if display:
        visualize_image(box_on_image)
    # Visualize reconstructed bounding boxes on depth in original scale
    orig_depth = sample['orig_depth'][idx, :, :, 0]
    orig_depth = flowlib.visualize_disp(orig_depth)
    box_on_depth = concat_box_on_image(orig_depth, boxes, color=(255, 255, 255))
    if display:
        visualize_image(box_on_depth)
    # Visualize reconstructed bounding boxes on flow in original scale
    orig_flow = sample['orig_flow'][idx, :, :, :]
    orig_flow = flowlib.visualize_flow(orig_flow)
    box_on_flow = concat_box_on_image(orig_flow.astype(np.uint8), boxes, color=(0, 0, 0))
    if display:
        visualize_image(box_on_flow)
    return box_on_image, box_on_depth, box_on_flow


def concat_images(images):
    # construct a full image containing all scale images
    max_im_width = 0
    for i in range(len(images)):
        im_width = images[i].shape[1]
        max_im_width = max(max_im_width, im_width)
    sum_im_height = 0
    for i in range(len(images)):
        im_height = images[i].shape[0]
        sum_im_height = sum_im_height + im_height
    if len(images[0].shape) == 2:
        image_all = np.zeros((sum_im_height, max_im_width))
        cnt = 0
        for i in range(len(images)):
            im = images[i]
            height, width = im.shape[0], im.shape[1]
            image_all[cnt:cnt + height, 0:width] = im
            cnt = cnt + height
    else:
        channel = images[0].shape[2]
        image_all = np.zeros((sum_im_height, max_im_width, channel))
        cnt = 0
        for i in range(len(images)):
            im = images[i]
            height, width = im.shape[0], im.shape[1]
            image_all[cnt:cnt + height, 0:width, :] = im
            cnt = cnt + height
    return image_all


def concat_box_on_image(image, boxes, color=(0, 255, 0)):
    if np.max(image) < 1.01:
        image = image * 255.0
    image = image.astype(np.uint8).copy()
    b, score = boxes[:, 0:4], boxes[:, 4]
    if b.shape[0] > 0:
        if np.max(b) < 1.01 and np.min(b) > -0.01:
            im_height, im_width = image.shape[0], image.shape[1]
            b[:, 0], b[:, 2] = b[:, 0] * im_width, b[:, 2] * im_width
            b[:, 1], b[:, 3] = b[:, 1] * im_height, b[:, 3] * im_height
        b = b.astype(np.int)
        for i in range(b.shape[0]):
            cv2.rectangle(image, (b[i, 0], b[i, 1]), (b[i, 2], b[i, 3]), color, 3)
            cv2.putText(image, '%.2f' % score[i], (b[i, 0] + 2, b[i, 1] + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    return image


def concat_heatmap_on_image(image, heatmap):
    im_height, im_width = image.shape[0], image.shape[1]
    heatmap = cv2.resize(heatmap, (im_width, im_height), interpolation=cv2.INTER_LINEAR)
    if len(heatmap.shape) == 2:
        heatmap = np.dstack((heatmap, heatmap, heatmap))
    if len(heatmap.shape) == 3 and heatmap.shape[2] == 1:
        heatmap = np.dstack((heatmap, heatmap, heatmap))
    heatmap_on_image = image * 0.2 + heatmap * 0.8
    return heatmap_on_image


def visualize_image(image):
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    plt.show()


def save_image(image, image_name):
    if np.max(image) <= 1.01:
        image = image * 255.0
    if not image.dtype == np.uint8:
        image = image.astype(np.uint8)
    image = Image.fromarray(image)
    image.save(image_name)
