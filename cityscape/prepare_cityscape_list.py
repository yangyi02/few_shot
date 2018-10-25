import os
import sys
import numpy as np

root_dir = '/mnt/project/yangyi05/CityScape'
image_root_dir = os.path.join(root_dir, 'leftImg8bit')
depth_root_dir = os.path.join(root_dir, 'disparity')
seg_root_dir = os.path.join(root_dir, 'gtFine')

status_list = ['train', 'val', 'test']
for status in status_list:
    lines = []
    image_dir = os.path.join(image_root_dir, status)
    folders = os.listdir(image_dir)
    for folder in folders:
        image_folder = os.path.join(image_dir, folder)
        image_files = os.listdir(image_folder)
        for image_file in image_files:
            if image_file.endswith('_leftImg8bit.png'):
                image_id = image_file.replace('_leftImg8bit.png', '')
                disparity_name = image_id + '_disparity.png'
                seg_name = image_id + '_gtFine_labelIds.png'
                lines.append(image_file + ' ' + disparity_name + ' ' + seg_name + '\n')
    print lines

    output_file = status + '.txt'
    handle = open(output_file, 'w')
    for line in lines:
        handle.writelines(line)
