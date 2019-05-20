import os
import sys
import json
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt 
import xml.etree.ElementTree as ET

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

def parse_xml(file):
    xml = ET.parse(file).getroot()
    box_all = []
    pts = ['xmin', 'ymin', 'xmax', 'ymax']

    # image size
    width = float(xml.find('size').find('width').text)
    height = float(xml.find('size').find('height').text)
    # original location of the chip
    chip_loc = []
    loc = xml.find('location')
    for i, pt in enumerate(pts):
        cur_pt = int(loc.find(pt).text) - 1
        chip_loc.append(cur_pt)

    # ratio
    width_ratio = (chip_loc[2] - chip_loc[0]) / width
    height_ratio = (chip_loc[3] - chip_loc[1]) / height

    # bounding boxes
    for obj in xml.iter('object'):
        name = obj.find('name').text.lower().strip()
        bbox = obj.find('bndbox')
        
        bndbox = []
        for i, pt in enumerate(pts):
            cur_pt = int(bbox.find(pt).text) - 1
            # scale to orginal size
            cur_pt = cur_pt * width_ratio if i % 2 == 0 else cur_pt * height_ratio
            cur_pt += chip_loc[0] if i % 2 == 0 else chip_loc[1]
            bndbox.append(cur_pt)
        bndbox.append(name)
        box_all += [bndbox]
    return box_all

def get_box_label(annos, imgid):
    valid_label = utils.TT100K_CLASSES
    img = annos["imgs"][imgid]
    box_all = []
    label_all = []
    for obj in img['objects']:
        box = obj['bbox']
        if obj['category'] in valid_label:
            box = [int(box['xmin']), int(box['ymin']), int(box['xmax']), int(box['ymax'])]
            box_all.append(np.clip(box, 0, 2047))
            label_all.append(obj['category'])
    return box_all, label_all


def _boxvis(img, gt_box_list):
    img1 = img.copy()
    mask = np.zeros_like(img1, dtype=np.uint8)
    chip_mask = np.zeros((30, 30, 1), dtype=np.uint8)
    vis_mask = np.zeros_like(img1, dtype=np.uint8)
            
    for box in gt_box_list:
        cv2.rectangle(img1, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 4)
        mask[box[1]:box[3], box[0]:box[2], :] = 255

        xmin, ymin, xmax, ymax = np.floor(box / 2048 * 30).astype(np.int)
        chip_mask[ymin : ymax+1, xmin : xmax+1] = 1
    
    mask_box = utils.generate_box_from_mask(chip_mask)
    for box in mask_box:
        xmin, ymin, xmax, ymax = np.floor(box / 30 * 2048).astype(np.int)
        vis_mask[ymin : ymax+1, xmin : xmax+1] = 255

    cv2.imwrite('vis1.jpg', img1)
    cv2.imwrite('vis2.png', vis_mask)

    plt.subplot(1, 2, 1); plt.imshow(img1[:, :, [2,1,0]])
    plt.subplot(1, 2, 2); plt.imshow(vis_mask)
    plt.show()
    cv2.waitKey()

def main():
    with open(src_annotation, 'r') as f:
        annos = json.load(f)

    with open(train_ids, 'r') as f:
        src_train_list = [x.strip() for x in f.readlines()]

    with open(os.path.join(list_dir, 'train.txt'), 'r') as f:
        chip_train_list = [x.strip() for x in f.readlines()]

    error_list = []
    for train_id in src_train_list:
        if(train_id == '10139'):
            print(train_id)
            box_all, label_all = get_box_label(annos, train_id)
            
            img = cv2.imread(os.path.join(src_traindir, train_id + '.jpg'))
            _boxvis(img, box_all)



if __name__ == '__main__':
    main()
