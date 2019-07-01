"""convert TT100k to VOC format
+ VOC2012
    + JPEGImages
    + SegmentationClass
"""

import os, sys
import glob
import cv2
import json
import shutil
import numpy as np
import concurrent.futures
import pdb

userhome = os.path.expanduser('~')
import anno_func

src_datadir = os.path.join(userhome, 'data/TT100K/data')
src_traindir = src_datadir + '/train'
src_testdir = src_datadir + '/test'
src_annotation = src_datadir + '/annotations.json'

dest_datadir = os.path.join(userhome, 'data/TT100K/TT100K_voc')
image_dir = dest_datadir + '/JPEGImages'
segmentation_dir = dest_datadir + '/SegmentationClass'
list_folder = dest_datadir + '/ImageSets'

if not os.path.exists(dest_datadir):
    os.mkdir(dest_datadir)
    os.mkdir(image_dir)
    os.mkdir(segmentation_dir)
    os.mkdir(list_folder)

# imageset list files
train_ids = src_traindir + '/ids.txt'
shutil.copy(train_ids, list_folder + '/train.txt')
test_ids = src_testdir + '/ids.txt'
shutil.copy(test_ids, list_folder + '/test.txt')

# copy train and test images
def _copy(src_image, dest_path):
    shutil.copy(src_image, dest_path)

def _resize(src_image, dest_path):
    img = cv2.imread(src_image)

    height, width = img.shape[:2]
    size = (int(width), int(height))

    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    name = os.path.basename(src_image)
    cv2.imwrite(os.path.join(dest_path, name), img)

def get_box(annos, imgid):
    img = annos["imgs"][imgid]
    box_all = []
    for obj in img['objects']:
        box = obj['bbox']
        box = [box['xmin'], box['ymin'], box['xmax'], box['ymax']]
        # box = [int(x * 0.3) for x in box]
        box_all.append(np.array(box) / 2048)
    return box_all

train_list = glob.glob(src_traindir + '/*.jpg')
test_list = glob.glob(src_testdir + '/*.jpg')
all_list = train_list + test_list

print('image....\n')
with concurrent.futures.ThreadPoolExecutor() as exector:
    exector.map(_resize, all_list, [image_dir]*len(all_list))

# mask
with open(src_annotation, 'r') as f:
    annos = json.load(f)
def _generate_mask(img_path):
    try:
        # image mask
        img_id = os.path.split(img_path)[-1][:-4]
        # im_data = anno_func.load_img(annos, src_datadir, img_id)
        # mask = anno_func.load_mask(annos, src_datadir, img_id, im_data)

        # height, width = mask.shape[:2]
        # size = (int(width), int(height))

        # mask = cv2.resize(mask, size)
        # maskname = os.path.join(segmentation_dir, img_id + '.png')
        # cv2.imwrite(maskname, mask)

        # chip mask 30x30
        mask_w, mask_h = 30, 30
        chip_mask = np.zeros((30, 30), dtype=np.uint8)
        boxes = get_box(annos, img_id)
        # for box in boxes:
        #     xmin, ymin, xmax, ymax = np.floor(box * 30).astype(np.int32)
        #     ignore_xmin = xmin - 1 if xmin - 1 >= 0 else 0
        #     ignore_ymin = ymin - 1 if ymin - 1 >= 0 else 0
        #     ignore_xmax = xmax + 1 if xmax + 1 < mask_w else mask_w - 1
        #     ignore_ymax = ymax + 1 if ymax + 1 < mask_h else mask_h - 1
        #     chip_mask[ignore_ymin : ignore_ymax+1, ignore_xmin : ignore_xmax+1] = 255
        for box in boxes:
            xmin, ymin, xmax, ymax = np.floor(box * 30).astype(np.int32)
            chip_mask[ymin : ymax+1, xmin : xmax+1] = 1
        maskname = os.path.join(segmentation_dir, img_id + '_chip.png')
        cv2.imwrite(maskname, chip_mask)

    except Exception as e:
        print(e)

print('mask...')
with concurrent.futures.ThreadPoolExecutor() as exector:
    exector.map(_generate_mask, all_list)
_generate_mask(all_list[0])

