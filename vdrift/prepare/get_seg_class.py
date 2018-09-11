import os
from PIL import Image
import numpy as np

seg_path = '/media/yi/DATA/data-orig/vdrift/seg'

seg_dict = dict()

seg_files = os.listdir(seg_path)
for seg_file in seg_files:
    seg_file_name = os.path.join(seg_path, seg_file)
    seg = np.array(Image.open(seg_file_name))
    seg1 = np.reshape(seg, (-1, 3))
    a, b, c = np.unique(seg1, return_inverse=True, return_counts=True, axis=0)
    # print a
    # print b
    # print c
    for i in range(a.shape[0]):
        value = tuple(a[i, :])
        if value not in seg_dict:
            seg_dict[value] = c[i]
        else:
            seg_dict[value] = seg_dict[value] + c[i]
    print seg_file_name
    print seg_dict
    # break

output_file = 'seg_class.txt'
with open(output_file, 'w') as handle:
    for key, value in seg_dict.iteritems():
        string = str(key) + ':' + str(value) + '\n'
        handle.writelines(string)
