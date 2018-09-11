import os
from PIL import Image
import numpy as np
import scipy.misc

seg_path = '/media/yi/DATA/data-orig/vdrift/seg'
seg_cls_path = '/media/yi/DATA/data-orig/vdrift/segcls'

seg_dict = dict()
seg_dict[(0 * 4 * 4 + 0 * 4 + 3)] = 0
seg_dict[(0 * 4 * 4 + 2 * 4 + 0)] = 1
seg_dict[(0 * 4 * 4 + 3 * 4 + 0)] = 2
seg_dict[(0 * 4 * 4 + 3 * 4 + 3)] = 3
seg_dict[(2 * 4 * 4 + 2 * 4 + 2)] = 4
seg_dict[(3 * 4 * 4 + 0 * 4 + 0)] = 5
seg_dict[(3 * 4 * 4 + 3 * 4 + 0)] = 6
seg_dict[(3 * 4 * 4 + 3 * 4 + 3)] = 7
seg_dict[(0 * 4 * 4 + 0 * 4 + 0)] = 1
seg_dict[(3 * 4 * 4 + 2 * 4 + 0)] = 6
inverse_seg_dict = dict()
inverse_seg_dict[0] = (0, 0, 255)
inverse_seg_dict[1] = (0, 128, 0)
inverse_seg_dict[2] = (0, 255, 0)
inverse_seg_dict[3] = (0, 255, 255)
inverse_seg_dict[4] = (128, 128, 128)
inverse_seg_dict[5] = (255, 0, 0)
inverse_seg_dict[6] = (255, 255, 0)
inverse_seg_dict[7] = (255, 255, 255)

seg_files = os.listdir(seg_path)
for seg_file in seg_files:
    print(seg_file)
    seg_file_name = os.path.join(seg_path, seg_file)
    seg = np.array(Image.open(seg_file_name))
    # seg1 = np.reshape(seg, (-1, 3))
    seg = seg / 64
    seg = seg[:, :, 0] * 4 * 4 + seg[:, :, 1] * 4 + seg[:, :, 2]
    seg_new = np.zeros_like(seg)
    for key in seg_dict.keys():
        mask = seg == key
        seg_new[mask] = seg_dict[key]
    seg_cls_file_name = os.path.join(seg_cls_path, seg_file)
    scipy.misc.imsave(seg_cls_file_name, seg_new)
    # break
