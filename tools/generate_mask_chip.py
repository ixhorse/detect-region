# -*- coding: utf-8 -*-
"""generate chip from segmentation mask
"""

import os, sys
import cv2
import json
import numpy as np
from tqdm import tqdm
from glob import glob
import utils
import pdb

home = os.path.expanduser('~')
root_datadir = os.path.join(home, 'data/TT100K')
src_traindir = root_datadir + '/data/train'
src_testdir = root_datadir + '/data/test'
src_annotation = root_datadir + '/data/annotations.json'

# TT100K imageset list files
train_ids = src_traindir + '/ids.txt'
test_ids = src_testdir + '/ids.txt'

dest_datadir = root_datadir + '/TT100K_chip_voc'
image_dir = dest_datadir + '/JPEGImages'
list_dir = dest_datadir + '/ImageSets/Main'
anno_dir = dest_datadir + '/Annotations'

mask_path = '../pytorch-deeplab-xception/run/mask'
            # 'codes/gluon-cv/projects/seg/outdir')

if not os.path.exists(dest_datadir):
    os.mkdir(dest_datadir)
    os.mkdir(image_dir)
    os.makedirs(list_dir)
    os.mkdir(anno_dir)

def main():
    with open(test_ids, 'r') as f:
        test_list = [x.strip() for x in f.readlines()]

    chip_loc = {}
    chip_name_list = []
    for imgid in tqdm(test_list):
        origin_img = cv2.imread(os.path.join(src_testdir, '%s.jpg'%imgid))
        mask_img = cv2.imread(os.path.join(mask_path, '%s.png'%imgid), cv2.IMREAD_GRAYSCALE)

        # mask_img = cv2.resize(mask_img, (2048, 2048), cv2.INTER_MAX)
        height, width = mask_img.shape[:2]
        # pdb.set_trace()
        chip_list = utils.region_box_generation(mask_img, (2048, 2048))
        # utils._boxvis(cv2.resize(mask_img, (2048, 2048)), chip_list, origin_img)
        # cv2.waitKey(0)

        for i, chip in enumerate(chip_list):
            chip_img = origin_img[chip[1]:chip[3], chip[0]:chip[2], :].copy()
            chip_img = cv2.resize(chip_img, (416, 416), cv2.INTER_AREA)
            chip_name = '%s_%d' % (imgid, i)
            cv2.imwrite(os.path.join(image_dir, '%s.jpg'%chip_name), chip_img)
            chip_name_list.append(chip_name)

            chip_info = {'loc': chip}
            chip_loc[chip_name] = chip_info

    # write test txt
    with open(os.path.join(list_dir, 'test.txt'), 'w') as f:
        f.writelines([x+'\n' for x in chip_name_list])
        print('write txt.')

    # write chip loc json
    with open(os.path.join(anno_dir, 'test_chip.json'), 'w') as f:
        json.dump(chip_loc, f)
        print('write json.')

if __name__ == '__main__':
    main()
