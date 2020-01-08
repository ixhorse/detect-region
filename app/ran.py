import os, sys
import cv2
import numpy as np

import torch

sys.path.append('../pytorch-deeplab-xception')
from modeling.deeplab import DeepLab

class RAN():
    def __init__(self, weight, gpu_ids):
        self.model = DeepLab(num_classes=2,
                            backbone='mobilenet',
                            output_stride=16)
        
        torch.cuda.set_device(gpu_ids)
        self.model = self.model.cuda()

        assert weight is not None
        if not os.path.isfile(weight):
            raise RuntimeError("=> no checkpoint found at '{}'" .format(weight))
        checkpoint = torch.load(weight)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()

        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img):
        # normalize
        img = cv2.resize(img, (480, 480))
        img = img.astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std
        img = img.transpose((2, 0, 1))
        img = img[np.newaxis, :, :, :]
        # to tensor
        img = torch.from_numpy(img).float().cuda()

        with torch.no_grad():
            output = self.model(img)
        return output
        