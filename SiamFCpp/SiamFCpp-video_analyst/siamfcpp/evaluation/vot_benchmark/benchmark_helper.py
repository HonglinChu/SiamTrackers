# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import json
from collections import OrderedDict
from os import listdir
from os.path import dirname, exists, isdir, join, realpath
from pathlib import Path

import cv2
import numpy as np


def get_json(path):
    with open(path) as f:
        return json.load(f)


def get_txt(path):
    with open(path) as f:
        return f.read()


def get_img(path):
    img = cv2.imread(path)
    return img


def get_files(path, suffix):
    if isinstance(path, str):
        p = Path(path)
    else:
        p = path
    list_dir = list(p.glob('*'))
    result = [x.name for x in list_dir if x.suffix == suffix]
    return result


def get_dataset_zoo():
    root = realpath(join(dirname(__file__), '../data'))
    zoos = listdir(root)

    def valid(x):
        y = join(root, x)
        if not isdir(y): return False

        return exists(join(y, 'list.txt')) \
               or exists(join(y, 'train', 'meta.json'))\
               or exists(join(y, 'ImageSets', '2016', 'val.txt'))

    zoos = list(filter(valid, zoos))
    return zoos


def load_dataset(vot_path, dataset):
    info = OrderedDict()
    if 'VOT' in dataset:
        base_path = join(vot_path, dataset)
        list_path = join(base_path, 'list.txt')
        f = get_txt(list_path)
        videos = [v.strip() for v in f.strip().split('\n')]
        for video in videos:
            video_path = join(base_path, video)
            image_path = join(video_path, 'color')
            if not exists(image_path):
                image_path = video_path
            image_files = sorted(get_files(image_path, '.jpg'))
            image_files = [join(image_path, x) for x in image_files]
            gt_path = join(video_path, 'groundtruth.txt')
            gt = get_txt(gt_path)
            gt = gt.strip().split('\n')

            gt = np.asarray([line.split(',') for line in gt], np.float32)

            if gt.shape[1] == 4:
                gt = np.column_stack(
                    (gt[:, 0], gt[:, 1], gt[:, 0], gt[:, 1] + gt[:, 3] - 1,
                     gt[:, 0] + gt[:, 2] - 1, gt[:, 1] + gt[:, 3] - 1,
                     gt[:, 0] + gt[:, 2] - 1, gt[:, 1]))
            info[video] = {'image_files': image_files, 'gt': gt, 'name': video}

    return info
