import glob
from collections import OrderedDict
from os.path import join

import numpy as np
from loguru import logger


def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])


def labelcolormap(N):
    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = 0
        g = 0
        b = 0
        id = i
        for j in range(7):
            str_id = uint82bin(id)
            r = r ^ (np.uint8(str_id[-1]) << (7 - j))
            g = g ^ (np.uint8(str_id[-2]) << (7 - j))
            b = b ^ (np.uint8(str_id[-3]) << (7 - j))
            id = id >> 3
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    return cmap


def label2color(cv2_gt, num=8):
    cmap = labelcolormap(num)
    [rows, cols, _] = cv2_gt.shape
    for i in range(rows):
        for j in range(cols):
            label = cv2_gt[i, j, 0]
            cv2_gt[i, j] = cmap[label]
    return cv2_gt


def load_dataset(davis_path, dataset):
    info = OrderedDict()

    if 'DAVIS' in dataset and 'TEST' not in dataset:
        #davis_path = '/data/data_track/DAVIS'
        list_path = join(davis_path, 'ImageSets', dataset[-4:], 'val.txt')
        with open(list_path) as f:
            videos = [v.strip() for v in f.readlines()]

        for video in videos:
            info[video] = {}
            if dataset[-4:] == '2017':
                info[video]['anno_files'] = sorted(
                    glob.glob(
                        join(davis_path, 'Annotations/480p', video, '*.png')))
            elif dataset[-4:] == '2016':
                info[video]['anno_files'] = sorted(
                    glob.glob(
                        join(davis_path, 'Annotations/480p_2016', video,
                             '*.png')))
            else:
                logger.error("{} is not supported".format(dataset))
                exit(-1)
            assert len(info[video]['anno_files']) > 0, logger.error(
                "no anno in path {}".format(
                    join(davis_path, 'Annotations/480p_2016', video)))

            info[video]['image_files'] = sorted(
                glob.glob(join(davis_path, 'JPEGImages/480p', video, '*.jpg')))
            info[video]['name'] = video

    return info


def MultiBatchIouMeter(thrs, outputs, targets, start=None, end=None):
    targets = np.array(targets)
    outputs = np.array(outputs)

    num_frame = targets.shape[0]
    if start is None:
        object_ids = np.array(list(range(outputs.shape[0]))) + 1
    else:
        object_ids = [int(id) for id in start]

    num_object = len(object_ids)
    res = np.zeros((num_object, len(thrs)), dtype=np.float32)

    output_max_id = np.argmax(outputs, axis=0).astype('uint8') + 1
    outputs_max = np.max(outputs, axis=0)
    for k, thr in enumerate(thrs):
        output_thr = outputs_max > thr
        for j in range(num_object):
            target_j = targets == object_ids[j]

            if start is None:
                start_frame, end_frame = 1, num_frame - 1
            else:
                start_frame, end_frame = start[str(object_ids[j])] + 1, end[str(
                    object_ids[j])] - 1
            iou = []
            for i in range(start_frame, end_frame):
                pred = (output_thr[i] * output_max_id[i]) == (j + 1)
                mask_sum = (pred == 1).astype(
                    np.uint8) + (target_j[i] > 0).astype(np.uint8)
                intxn = np.sum(mask_sum == 2)
                union = np.sum(mask_sum > 0)
                if union > 0:
                    iou.append(intxn / union)
                elif union == 0 and intxn == 0:
                    iou.append(1)
            res[j, k] = np.mean(iou)
    return res
