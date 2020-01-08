import os, sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch

from ran import RAN
from rbg import RBG
from general_detector import generalDetector

class smallDetector():
    def __init__(self):
        ran_weight = '/home/mcc/working/detect-region/pytorch-deeplab-xception/run/tt100k/deeplab-mobile-region/experiment_33/checkpoint_122.pth.tar'
        self.ran = RAN(weight=ran_weight, gpu_ids=0)
        self.rbg = RBG()
        self.gdet = generalDetector()

    def inference(self, img_path):
        self.img = cv2.imread(img_path)
        output = self.ran.inference(self.img)
        preds = output.data.cpu().numpy()
        self.mask = np.argmax(preds, axis=1)[0]
        self.show_result(self.img, self.mask)
        return self.mask

    def show_result(self, img, pred):
        plt.figure(figsize=(10, 10))
        plt.subplot(1, 2, 1).imshow(img[:, :, ::-1])
        plt.subplot(1, 2, 2).imshow(pred.astype(np.uint8) * 255)
        plt.show()
        cv2.waitKey()


if __name__ == "__main__":
    img_path = '/home/mcc/data/TT100K/region_voc/JPEGImages/94910.jpg'
    detector = smallDetector()
    detector.inference(img_path)