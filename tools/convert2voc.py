"""convert TT100k to VOC format
+ VOC2012
    + JPEGImages
    + SegmentationClass
"""

import os, sys
import glob
import cv2
import numpy as np
import shutil
import json
import concurrent.futures
import pdb

sys.path.append('./code/python')
import anno_func

src_datadir = './data'
src_traindir = src_datadir + '/train'
src_testdir = src_datadir + '/test'
src_annotation = src_datadir + '/annotations.json'

dest_datadir = './TT100K_voc'
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
shutil.copy(test_ids, list_folder + '/val.txt')

# copy train and test images
def _copy(src_image, dest_path):
    shutil.copy(src_image, dest_path)

def _resize(src_image, dest_path):
    img = cv2.imread(src_image)

    height, width = img.shape[:2]
    size = (int(width*0.3), int(height*0.3))

    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    name = os.path.basename(src_image)
    cv2.imwrite(os.path.join(dest_path, name), img)

train_list = glob.glob(src_traindir + '/*.jpg')
test_list = glob.glob(src_testdir + '/*.jpg')
all_list = train_list + test_list

# pdb.set_trace()

with concurrent.futures.ProcessPoolExecutor() as exector:
    exector.map(_resize, all_list, [image_dir]*len(all_list))

# mask
with open(src_annotation, 'r') as f:
    annos = json.load(f)
def _generate_mask(img_path):
    try:
        img_id = os.path.split(img_path)[-1][:-4]
        im_data = anno_func.load_img(annos, src_datadir, img_id)
        mask = anno_func.load_mask(annos, src_datadir, img_id, im_data)

        height, width = mask.shape[:2]
        size = (int(width*0.3), int(height*0.3))

        mask = cv2.resize(mask, size)
        filename = os.path.join(segmentation_dir, img_id + '.png')
        cv2.imwrite(filename, mask)
    except Exception as e:
        print(e)

with concurrent.futures.ProcessPoolExecutor() as exector:
    exector.map(_generate_mask, all_list)