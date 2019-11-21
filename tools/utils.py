import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

TT100K_CLASSES = (
    'p11', 'pl5', 'pne', 'il60', 'pl80', 'pl100', 'il80', 'po', 'w55',
    'pl40', 'pn', 'pm55', 'w32', 'pl20', 'p27', 'p26', 'p12', 'i5',
    'pl120', 'pl60', 'pl30', 'pl70', 'pl50', 'ip', 'pg', 'p10', 'io',
    'pr40', 'p5', 'p3', 'i2', 'i4', 'ph4', 'wo', 'pm30', 'ph5', 'p23',
    'pm20', 'w57', 'w13', 'p19', 'w59', 'il100', 'p6', 'ph4.5')

def generate_box_from_mask(mask):
    """
    Args:
        mask: 0/1 array
    """
    box_all = []
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        x, y, w, h = cv.boundingRect(contours[i])
#if w < 2 and h < 2:
#           continue
        box_all.append([x, y, x+w, y+h])
    return np.array(box_all)


def enlarge_box(mask_box, image_size, ratio=2):
    """
    Args:
        mask_box: list of box
        image_size: (width, height)
        ratio: int
    """
    new_mask_box = []
    for box in mask_box:
        w = box[2] - box[0]
        h = box[3] - box[1]
        center_x = w / 2 + box[0]
        center_y = h / 2 + box[1]
        w = w * ratio / 2
        h = h * ratio / 2
        new_box = [center_x-w if center_x-w > 0 else 0,
                    center_y-h if center_y-h > 0 else 0,
                    center_x+w if center_x+w < image_size[0] else image_size[0]-1,
                    center_y+h if center_y+h < image_size[1] else image_size[1]-1]
        new_box = [int(x) for x in new_box]
        new_mask_box.append(new_box)
    return new_mask_box


def resize_box(box, original_size, dest_size):
    """
    Args:
        box: [xmin, ymin, xmax, ymax]
        original_size: int
        dest_size: int
    """
    ratio = dest_size / original_size
    box = np.array(box) * ratio
    return list(np.clip(box, 0, dest_size-1).astype(np.int32))


def overlap(box1, box2, thresh = 0.75):
    """ (box1 \cup box2) / box2
    Args:
        box1: [xmin, ymin, xmax, ymax]
        box2: [xmin, ymin, xmax, ymax]
    """
    matric = np.array([box1, box2])
    u_xmin = np.max(matric[:,0])
    u_ymin = np.max(matric[:,1])
    u_xmax = np.min(matric[:,2])
    u_ymax = np.min(matric[:,3])
    u_w = u_xmax - u_xmin
    u_h = u_ymax - u_ymin
    if u_w <= 0 or u_h <= 0:
        return False
    u_area = u_w * u_h
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    if u_area / box2_area < thresh:
        return False
    else:
        return True

def _boxvis(img, box_list, origin_img=None, binary=True):
    # if binary:
    #     ret, img = cv.threshold(img, 0, 255, cv.THRESH_BINARY)
    for box in box_list:
        cv.rectangle(img, (box[0], box[1]), (box[2], box[3]), 255, 4)
    plt.subplot(1, 2, 1); plt.imshow(img, cmap='gray')
    if not origin_img is None:
        for box in box_list:
            cv.rectangle(origin_img, (box[0], box[1]), (box[2], box[3]), 255, 4)
        plt.subplot(1, 2, 2); plt.imshow(origin_img[:, :, [2,1,0]])
    plt.show()
    # cv.namedWindow('a', cv.WINDOW_AUTOSIZE)
    # cv.imshow('a', binary)
    # key = cv.waitKey(0)
    # sys.exit(0)

def iou_calc1(boxes1, boxes2):
    """
    array
    :param boxes1: boxes1和boxes2的shape可以不相同，但是需要满足广播机制
    :param boxes2: 且需要保证最后一维为坐标维，以及坐标的存储结构为(xmin, ymin, xmax, ymax)
    :return: 返回boxes1和boxes2的IOU，IOU的shape为boxes1和boxes2广播后的shape[:-1]
    """
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    # 计算出boxes1和boxes2相交部分的左上角坐标、右下角坐标
    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    # 计算出boxes1和boxes2相交部分的宽、高
    # 因为两个boxes没有交集时，(right_down - left_up) < 0，所以maximum可以保证当两个boxes没有交集时，它们之间的iou为0
    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + 1e-16 + boxes2_area - inter_area
    IOU = 1.0 * inter_area / union_area
    return IOU


def iou_calc2(boxes1, boxes2):
    """
    array
    :param boxes1: boxes1和boxes2的shape可以不相同，但是需要满足广播机制
    :param boxes2: 且需要保证最后一维为坐标维，以及坐标的存储结构为(xmin, ymin, xmax, ymax)
    :return: 返回boxes1和boxes2的IOU，IOU的shape为boxes1和boxes2广播后的shape[:-1]
    """
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    # 计算出boxes1和boxes2相交部分的左上角坐标、右下角坐标
    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    # 计算出boxes1和boxes2相交部分的宽、高
    # 因为两个boxes没有交集时，(right_down - left_up) < 0，所以maximum可以保证当两个boxes没有交集时，它们之间的iou为0
    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    IOU = 1.0 * inter_area / boxes2_area
    return IOU


def nms(prediction, score_threshold=0.005, iou_threshold=0.5, overlap_threshold=0.95):
    """
    :param prediction:
    (x, y, w, h, conf, cls)
    :return: best_bboxes
    """
    prediction = np.array(prediction)
    detections = prediction[(-prediction[:,4]).argsort()]
    # Iterate through all predicted classes
    unique_labels = np.unique(detections[:, -1])

    best_bboxes = []
    for cls in unique_labels:
        cls_mask = (detections[:, 5] == cls)
        cls_bboxes = detections[cls_mask]

        # python code
        while len(cls_bboxes) > 0:
            best_bbox = cls_bboxes[0]
            best_bboxes.append(best_bbox)
            cls_bboxes = cls_bboxes[1:]
            # iou
            iou = iou_calc1(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            iou_mask = iou > iou_threshold
             # overlap
            overlap = iou_calc2(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            overlap_mask = overlap > overlap_threshold

            weight = np.ones((len(iou),), dtype=np.float32)
            weight[iou_mask] = 0.0
            weight[overlap_mask] = 0.0
           
            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > score_threshold
            cls_bboxes = cls_bboxes[score_mask]
    best_bboxes = np.array(best_bboxes)
    return best_bboxes