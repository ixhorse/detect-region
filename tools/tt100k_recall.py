"""
recall of segmentation result of tt100k
"""

import os, sys
import numpy as np
import cv2 as cv
from glob import glob
import json
from tqdm import tqdm
from operator import mul
import utils
import pdb


user_home = os.path.expanduser('~')
datadir = os.path.join(user_home, 'data/TT100K')
label_path = datadir + '/TT100K_voc/SegmentationClass'
annos_path = datadir + '/data/annotations.json'
image_path = datadir + '/TT100K_voc/JPEGImages'
mask_path = \
    '../pytorch-deeplab-xception/run/mask'
    #'../../gluon-cv/projects/seg/outdir'



def get_box(annos, imgid):
    img = annos["imgs"][imgid]
    box_all = []
    for obj in img['objects']:
        box = obj['bbox']
        box = [int(box['xmin']), int(box['ymin']), int(box['xmax']), int(box['ymax'])]
        # box = [int(x * 0.3) for x in box]
        box_all.append(box)
    return box_all


def _boxvis(mask, mask_box):
    ret, binary = cv.threshold(mask, 0, 255, cv.THRESH_BINARY)
    print(mask_box)
    for box in mask_box:
        cv.rectangle(binary, (box[0], box[1]), (box[2], box[3]), 100, 2)
    cv.imshow('a', binary)
    key = cv.waitKey(0)
    sys.exit(0)


def vis_undetected_image(img_list):
    annos = json.loads(open(annos_path).read())

    for image in img_list:
        mask_file = os.path.join(mask_path, image+'.png')
        image_file = os.path.join(image_path, image+'.jpg')

        mask_img = cv.imread(mask_file, cv.IMREAD_GRAYSCALE)
        original_img = cv.imread(image_file)
        original_img[:,:,1] = np.clip(original_img[:,:,1] + mask_img*70, 0, 255)

        label_box = get_box(annos, image)
        for box in label_box:
            cv.rectangle(original_img, (box[0], box[1]), (box[2], box[3]), (0,0,255), 1, 1)

        cv.imshow('1', original_img)
        key = cv.waitKey(1000*100)
        if key == 27:
            break

def main():
    annos = json.loads(open(annos_path).read())

    label_object = []
    detect_object = []
    mask_object = []
    undetected_img = []
    for raw_file in tqdm(glob(mask_path + '/*.png')):
        img_name = os.path.basename(raw_file)
        imgid = os.path.splitext(img_name)[0]
        label_file = os.path.join(label_path, img_name)
        image_file = os.path.join(image_path, imgid + '.jpg')

        mask_img = cv.imread(raw_file, cv.IMREAD_GRAYSCALE)
        # mask_img = cv.resize(mask_img, (2048, 2048), interpolation=cv.INTER_LINEAR)

        height, width = mask_img.shape[:2]

        label_box = get_box(annos, imgid)
        mask_box = utils.generate_box_from_mask(mask_img)
        mask_box = list(map(utils.resize_box, mask_box,
                        [width]*len(mask_box), [2048]*len(mask_box)))
        mask_box = utils.enlarge_box(mask_box, (2048, 2048), ratio=2)
        # _boxvis(mask_img, mask_box)
        # break

        count = 0
        for box1 in label_box:
            for box2 in mask_box:
                if utils.overlap(box2, box1):
                    count += 1
                    break

        label_object.append(len(label_box))
        detect_object.append(count)
        mask_object.append(len(mask_box))
        if len(label_box) != count:
            undetected_img.append(imgid)

    print('recall: %f' % (np.sum(detect_object) / np.sum(label_object)))
    print('detect box avg: %f, std %d' %(np.mean(mask_object), np.std(mask_object)))
    # print(undetected_img)

if __name__ == '__main__':
    img_list = ['80124', '91876', '4867', '1883', '97776', '43600', '91027', '31998', '25647', '3448', '33807', '25503', '42804', '75545', '13802', '68191', '41889', '83208', '51875', '52429', '25333', '98318', '69666', '6704', '40008', '29597', '28256', '48397', '58030', '44609', '49063', '18935', '36686', '5886', '36140', '59980', '75883', '27121', '52988', '87970', '56922', '2257', '43792', '73624', '96224', '14724', '6599', '87927', '5332', '75881', '96754', '62161', '31752', '37771', '48123', '10175', '88185', '34636', '4902', '1792', '49013', '81686', '22599', '89739', '42891', '29107', '12467', '75952', '79441', '40363', '76771', '90940', '17892', '60075', '93350', '2736', '79572', '40799', '39950', '58781', '42905', '77713', '44235', '94686', '1891', '2777', '86368', '94893', '89987', '90411', '94297', '64408', '51178', '63436', '98790', '71249', '41372', '92518', '29178', '4232', '39610', '76122', '33871', '74553', '58281', '80925', '62728', '38273', '74148', '58006', '7219', '55926', '66398', '68066', '69108', '13919', '76568', '76474', '74726', '35422', '94307', '97409', '49054', '1747', '12339', '36412', '15237', '16395', '72200', '73250', '46346', '52875', '59076', '46674', '25839', '27083', '34623', '59240', '38199', '17589', '13931', '8783', '84252', '29297', '86217', '13', '26672', '22075', '52252', '71017', '96501', '66056', '69425', '33714', '84327', '7103', '2604', '23320', '39338', '86066', '27663', '96461', '4001', '8099', '9401', '64497', '85758', '74315', '52240', '76329', '15332', '62157', '60415', '3913', '74518', '86340', '14558', '68231', '26065', '62713', '94966', '90420', '97811', '92707', '26234', '76837', '17947', '74668', '41988', '42661']
    # vis_undetected_image(img_list)
    main()
