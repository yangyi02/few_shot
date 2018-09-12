import os
import numpy as np


def nms(score_maps, offsets, im_height, im_width, overlap_thresh=0.5):
    boxes = score2box(score_maps, offsets, im_height, im_width)
    boxes = non_max_suppression_fast(boxes, overlap_thresh)
    return boxes


def score2box(score_maps, offsets, im_height, im_width, thresh=0.1):
    """
    Transfer score maps with offsets to boxes
    """
    boxes = np.zeros((0, 5))
    for i in range(len(score_maps)):
        of = offsets[i][0, :, :, :].transpose(1, 2, 0)
        sc = score_maps[i][0, 0, :, :]
        ou_height, ou_width = sc.shape[0], sc.shape[1]
        b_idx = np.nonzero(sc >= thresh)
        y_c, x_c = b_idx[0], b_idx[1]
        b = np.zeros((len(b_idx[0]), 5))
        for k in range(b.shape[0]):
            offset = of[y_c[k], x_c[k], :]
            b[k, 0] = x_c[k] * 1.0 / ou_width + offset[0]
            b[k, 1] = y_c[k] * 1.0 / ou_height + offset[1]
            b[k, 2] = x_c[k] * 1.0 / ou_width + offset[2]
            b[k, 3] = y_c[k] * 1.0 / ou_height + offset[3]
            b[k, 4] = sc[y_c[k], x_c[k]]
        boxes = np.concatenate((boxes, b))
    boxes = np.maximum(np.minimum(boxes, 1.0), 0.0)
    boxes[:, 0], boxes[:, 2] = boxes[:, 0] * im_width, boxes[:, 2] * im_width
    boxes[:, 1], boxes[:, 3] = boxes[:, 1] * im_height, boxes[:, 3] * im_height
    return boxes


# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    sc = boxes[:,4]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(sc)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick]
