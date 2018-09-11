import numpy
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as cl
import png
from PIL import Image


def read_disp_png(disp_file):
    """
    Read disparity from KITTI .png file
    :param disp_file: name of the disp file
    :return: disparity data in matrix
    """
    disp = numpy.array(Image.open(disp_file))
    disp = disp.astype(numpy.float32) / 256.0
    return disp


def visualize_disp(disp):
    disp = (disp - numpy.min(disp)) / (numpy.max(disp) - numpy.min(disp))
    cmap = plt.get_cmap('gist_rainbow')
    disp = cmap(disp)
    disp = disp[:, :, 0:3]
    return disp


def visualize_flow(flow, max_flow=None, mode='Y'):
    """
    this function visualize the input flow
    :param flow: input flow in array
    :param max_flow: maximum flow
    :param mode: choose which color mode to visualize the flow (Y: Ccbcr, RGB: RGB color)
    :return: None
    """
    if mode == 'Y':
        # Ccbcr color wheel
        img = flow_to_image(flow, max_flow)
    elif mode == 'RGB':
        (h, w) = flow.shape[0:2]
        du = flow[:, :, 0]
        dv = flow[:, :, 1]
        valid = flow[:, :, 2]
        if max_flow is None:
            max_flow = max(numpy.max(du), numpy.max(dv))
        img = numpy.zeros((h, w, 3), dtype=numpy.float64)
        # angle layer
        img[:, :, 0] = numpy.arctan2(dv, du) / (2 * numpy.pi)
        # magnitude layer, normalized to 1
        img[:, :, 1] = numpy.sqrt(du * du + dv * dv) * 8 / max_flow
        # phase layer
        img[:, :, 2] = 8 - img[:, :, 1]
        # clip to [0,1]
        small_idx = img[:, :, 0:3] < 0
        large_idx = img[:, :, 0:3] > 1
        img[small_idx] = 0
        img[large_idx] = 1
        # convert to rgb
        img = cl.hsv_to_rgb(img)
        # remove invalid point
        img[:, :, 0] = img[:, :, 0] * valid
        img[:, :, 1] = img[:, :, 1] * valid
        img[:, :, 2] = img[:, :, 2] * valid

    return img


def flow_to_image(flow, max_flow=None):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    UNKNOWN_FLOW_THRESH = 1e7

    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, numpy.max(u))
    minu = min(minu, numpy.min(u))

    maxv = max(maxv, numpy.max(v))
    minv = min(minv, numpy.min(v))

    if max_flow is None:
        rad = numpy.sqrt(u ** 2 + v ** 2)
        maxrad = max(-1, numpy.max(rad))
    else:
        maxrad = numpy.sqrt(max_flow ** 2 + max_flow ** 2)

    print("max flow: %.4f\nflow range:\nu = %.3f .. %.3f\nv = %.3f .. %.3f" % (maxrad, minu, maxu, minv, maxv))

    u = u/(maxrad + numpy.finfo(float).eps)
    v = v/(maxrad + numpy.finfo(float).eps)

    img = compute_color(u, v)

    idx = numpy.repeat(idxUnknow[:, :, numpy.newaxis], 3, axis=2)
    img[idx] = 0

    return numpy.uint8(img)


def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = numpy.zeros([h, w, 3])
    nanIdx = numpy.isnan(u) | numpy.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = numpy.size(colorwheel, 0)

    rad = numpy.sqrt(u**2+v**2)

    a = numpy.arctan2(-v, -u) / numpy.pi

    fk = (a+1) / 2 * (ncols - 1) + 1

    k0 = numpy.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0

    for i in range(0, numpy.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = numpy.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = numpy.uint8(numpy.floor(255 * col*(1-nanIdx)))

    return img


def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = numpy.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = numpy.transpose(numpy.floor(255*numpy.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - numpy.transpose(numpy.floor(255*numpy.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = numpy.transpose(numpy.floor(255*numpy.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - numpy.transpose(numpy.floor(255*numpy.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = numpy.transpose(numpy.floor(255*numpy.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - numpy.transpose(numpy.floor(255 * numpy.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255

    return colorwheel


def read_flow(filename):
    """
    read optical flow from Middlebury .flo file
    :param filename: name of the flow file
    :return: optical flow data in matrix
    """
    f = open(filename, 'rb')
    magic = numpy.fromfile(f, numpy.float32, count=1)
    data2d = None

    if 202021.25 != magic:
        print('Magic number incorrect. Invalid .flo file')
    else:
        w = numpy.fromfile(f, numpy.int32, count=1)
        h = numpy.fromfile(f, numpy.int32, count=1)
        print("Reading %d x %d flo file" % (h, w))
        data2d = numpy.fromfile(f, numpy.float32, count=2 * w * h)
        # reshape data into 3D array (columns, rows, channels)
        data2d = numpy.resize(data2d, (h[0], w[0], 2))
    f.close()
    return data2d


def read_flow_png(flow_file):
    """
    Read optical flow from KITTI .png file
    :param flow_file: name of the flow file
    :return: optical flow data in matrix
    """
    flow_object = png.Reader(filename=flow_file)
    flow_direct = flow_object.asDirect()
    flow_data = list(flow_direct[2])
    (w, h) = flow_direct[3]['size']
    flow = numpy.zeros((h, w, 3), dtype=numpy.float64)
    for i in range(len(flow_data)):
        flow[i, :, 0] = flow_data[i][0::3]
        flow[i, :, 1] = flow_data[i][1::3]
        flow[i, :, 2] = flow_data[i][2::3]

    invalid_idx = (flow[:, :, 2] == 0)
    flow[:, :, 0:2] = (flow[:, :, 0:2] - 2 ** 15) / 64.0
    flow[invalid_idx, 0] = 0
    flow[invalid_idx, 1] = 0
    return flow


def write_flow_png(flo, flow_file):
    h, w, _ = flo.shape
    out_flo = numpy.ones((h, w, 3), dtype=numpy.float32)
    out_flo[:,:,0] = numpy.maximum(numpy.minimum(flo[:,:,0]*64.0 + 2**15, 2**16-1), 0)
    out_flo[:,:,1] = numpy.maximum(numpy.minimum(flo[:,:,1]*64.0 + 2**15, 2**16-1), 0)
    out_flo = out_flo.astype(numpy.uint16)

    with open(flow_file, 'wb') as f:
        writer = png.Writer(width=w, height=h, bitdepth=16)
# Convert z to the Python list of lists expected by
# the png writer.
        z2list = out_flo.reshape(-1, w*3).tolist()
        writer.write(f, z2list)


def write_flow(flow, filename):
    """
    write optical flow in Middlebury .flo format
    :param flow: optical flow map
    :param filename: optical flow file path to be saved
    :return: None
    """
    f = open(filename, 'wb')
    magic = numpy.array([202021.25], dtype=numpy.float32)
    (height, width) = flow.shape[0:2]
    w = numpy.array([width], dtype=numpy.int32)
    h = numpy.array([height], dtype=numpy.int32)
    magic.tofile(f)
    w.tofile(f)
    h.tofile(f)
    flow.tofile(f)
    f.close()

