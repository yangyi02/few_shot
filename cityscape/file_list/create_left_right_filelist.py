import os
import sys
import numpy as np
from random import shuffle

# cityscape_dir = '/mnt/project/yangyi05/CityScape'
cityscape_dir = '/media/yi/DATA/data-orig/CityScape'
left_image_name = 'leftImg8bit'
right_image_name = 'rightImg8bit'
disparity_name= 'disp_unsup'
segmentation_name = 'gtFine'
left_image_dir = os.path.join(cityscape_dir, left_image_name)
right_image_dir = os.path.join(cityscape_dir, right_image_name)
disparity_dir = os.path.join(cityscape_dir, disparity_name)
segmentation_dir = os.path.join(cityscape_dir, segmentation_name)
status_list = ['train', 'val', 'test']

for status in status_list:
    output_file = 'left_right_' + status + '_files.txt'
    handle = open(output_file, 'w')
    left_image_folder = os.path.join(left_image_dir, status)
    right_image_folder = os.path.join(right_image_dir, status)
    disparity_folder = os.path.join(disparity_dir, status)
    segmentation_folder = os.path.join(segmentation_dir, status)
    folders = os.listdir(left_image_folder)
    for folder in folders:
        left_folder = os.path.join(left_image_folder, folder)
        left_file_names = os.listdir(left_folder)
        left_file_names = sorted(left_file_names)
        print len(left_file_names)
        right_folder = os.path.join(right_image_folder, folder)
        right_file_names = os.listdir(right_folder)
        right_file_names = sorted(right_file_names)
        print len(right_file_names)
        dispa_folder = os.path.join(disparity_folder, folder)
        dispa_file_names = os.listdir(dispa_folder)
        dispa_file_names = sorted(dispa_file_names)
        segme_folder = os.path.join(segmentation_folder, folder)
        segme_file_names = os.listdir(segme_folder)
        segme_file_names = sorted(segme_file_names)
        for i in range(len(segme_file_names)):
                if segme_file_names[i].endswith('_gtFine_labelIds.png'):
                    file_names = segme_file_names[i].split('_')
                    file_id = file_names[0] + '_' + file_names[1] + '_' + file_names[2]
                    f_l = os.path.join(left_folder, file_id + '_leftImg8bit.png')
                    f_r = os.path.join(right_folder, file_id + '_rightImg8bit.png')
                    di = os.path.join(dispa_folder, file_id + '_disparity.png')
                    se = os.path.join(segme_folder, segme_file_names[i])
                    string = f_l + ' ' + f_r + ' ' + se + '\n'
                    print(string)
                    handle.writelines(string)
                    # sys.exit()

# # Random shuffle files
# for status in status_list:
#     input_file = 'cityscape_' + status + '_files.txt'
#     lines = open(input_file).readlines()
#     shuffle(lines)
#     output_file = 'cityscape_' + status + '_files_shuffle.txt'
#     handle = open(output_file, 'w')
#     for line in lines:
#         handle.writelines(line)
