import numpy as np


def nms(heatmaps, offsets, im_height, im_width, overlap_thresh=0.5):
    boxes = heatmaps2boxes(heatmaps, offsets, im_height, im_width)
    for i in range(len(boxes)):
        boxes[i] = non_max_suppression_fast(boxes[i], overlap_thresh)
    return boxes


def heatmap2box(heatmaps, offsets, score_thresh=0.001):
    """
    Transfer heat maps with offsets to boxes for an image
    Box coordinate in [0, 1]
    """
    box = np.zeros((0, 5))
    for i in range(len(heatmaps)):
        if len(heatmaps[i].shape) == 3:
            heatmap = heatmaps[i][:, :, 0]
        else:
            heatmap = heatmaps[i]
        offset = offsets[i]
        ou_height, ou_width = heatmap.shape[0], heatmap.shape[1]
        [y_c, x_c] = np.nonzero(heatmap >= score_thresh)
        b = np.zeros((len(y_c), 5))
        for k in range(b.shape[0]):
            of = offset[y_c[k], x_c[k], :]
            b[k, 0] = x_c[k] * 1.0 / ou_width + of[0]
            b[k, 1] = y_c[k] * 1.0 / ou_height + of[1]
            b[k, 2] = x_c[k] * 1.0 / ou_width + of[2]
            b[k, 3] = y_c[k] * 1.0 / ou_height + of[3]
            b[k, 4] = heatmap[y_c[k], x_c[k]]
        box = np.concatenate((box, b))
    box = np.maximum(np.minimum(box, 1.0), 0.0)
    return box


def heatmaps2boxes(heatmaps, offsets, im_height, im_width, score_thresh=0.1):
    """
    Transfer heat maps with offsets to boxes for a full batch
    Box coordinate in [0, im_height/im_width]
    """
    boxes = []
    batch_size = heatmaps[0].shape[0]
    for n in range(batch_size):
        box = np.zeros((0, 5))
        for i in range(len(heatmaps)):
            heatmap = heatmaps[i][n, :, :, 0]
            offset = offsets[i][n, :, :, :]
            ou_height, ou_width = heatmap.shape[0], heatmap.shape[1]
            [y_c, x_c] = np.nonzero(heatmap >= score_thresh)
            b = np.zeros((len(y_c), 5))
            for k in range(b.shape[0]):
                of = offset[y_c[k], x_c[k], :]
                b[k, 0] = x_c[k] * 1.0 / ou_width + of[0]
                b[k, 1] = y_c[k] * 1.0 / ou_height + of[1]
                b[k, 2] = x_c[k] * 1.0 / ou_width + of[2]
                b[k, 3] = y_c[k] * 1.0 / ou_height + of[3]
                b[k, 4] = heatmap[y_c[k], x_c[k]]
            box = np.concatenate((box, b))
        box = np.maximum(np.minimum(box, 1.0), 0.0)
        box[:, 0], box[:, 2] = box[:, 0] * im_width, box[:, 2] * im_width
        box[:, 1], box[:, 3] = box[:, 1] * im_height, box[:, 3] * im_height
        boxes.append(box)
    return boxes


# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return np.zeros((0, 5))

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
