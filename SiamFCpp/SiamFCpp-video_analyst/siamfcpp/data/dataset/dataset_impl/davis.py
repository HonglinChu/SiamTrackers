# import glob

# from .utils import Dataset

# -*- coding: utf-8 -*-
import os
import os.path as osp
import pickle
from collections import defaultdict

import cv2
import numpy as np
from loguru import logger
from PIL import Image

from siamfcpp.data.dataset.dataset_base import (TRACK_DATASETS,
                                                    VOS_DATASETS, DatasetBase)
from siamfcpp.pipeline.utils.bbox import xywh2xyxy


@VOS_DATASETS.register
class DavisDataset(DatasetBase):
    r"""
    COCO dataset helper
    Hyper-parameters
    ----------------
    dataset_root: str
        path to root of the dataset
    subsets: list
        dataset split name list [train2017, val2017, train2016]
    ratio: float
        dataset ratio. used by sampler (data.sampler).
    max_diff: int
        maximum difference in index of a pair of sampled frames 
    """
    data_items = []

    default_hyper_params = dict(
        dataset_root="datasets/DAVIS",
        subsets=[
            "train2017",
        ],
        ratio=1.0,
        max_diff=10,
    )

    def __init__(self) -> None:
        r"""
        Create davis dataset 
        """
        super(DavisDataset, self).__init__()
        self._state["dataset"] = None

    def update_params(self):
        r"""
        an interface for update params
        """
        dataset_root = self._hyper_params["dataset_root"]
        self._hyper_params["dataset_root"] = osp.realpath(dataset_root)
        if len(DavisDataset.data_items) == 0:
            self._ensure_cache()

    def __getitem__(self, item):
        """
        :param item: int, video id
        :return:
            image_files
            annos
            meta (optional)
        """
        record = DavisDataset.data_items[item]
        anno = [[anno_file, record['obj_id']] for anno_file in record["annos"]]
        sequence_data = dict(image=record["image_files"], anno=anno)
        return sequence_data

    def __len__(self):
        return len(DavisDataset.data_items)

    def _ensure_cache(self):
        dataset_root = self._hyper_params["dataset_root"]
        for subset in self._hyper_params["subsets"]:
            year = subset[-4:]
            image_root = osp.join(dataset_root, "JPEGImages", "480p")
            if year == "2016":
                anno_root = osp.join(dataset_root, "Annotations", "480p_2016")
            else:
                anno_root = osp.join(dataset_root, "Annotations", "480p")
            data_anno_list = []
            cache_file = osp.join(dataset_root, "cache/{}.pkl".format(subset))
            if osp.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    DavisDataset.data_items += pickle.load(f)
                logger.info("{}: loaded cache file {}".format(
                    DavisDataset.__name__, cache_file))
            else:
                meta_file = osp.join(dataset_root, "ImageSets", year,
                                     subset[:-4] + ".txt")
                with open(meta_file) as f:
                    video_names = [item.strip() for item in f.readlines()]
                for video_name in video_names:
                    img_dir = os.path.join(image_root, video_name)
                    anno_dir = os.path.join(anno_root, video_name)
                    object_dict = defaultdict(list)
                    for anno_name in os.listdir(anno_dir):
                        anno_file = os.path.join(anno_dir, anno_name)
                        anno_data = np.array(Image.open(anno_file),
                                             dtype=np.uint8)
                        obj_ids = np.unique(anno_data)
                        for obj_id in obj_ids:
                            if obj_id > 0:
                                object_dict[obj_id].append(
                                    anno_name.split(".")[0])
                    for k, v in object_dict.items():
                        record = {}
                        record["obj_id"] = k
                        record["image_files"] = [
                            osp.join(img_dir, frame + '.jpg') for frame in v
                        ]
                        record["annos"] = [
                            osp.join(anno_dir, frame + '.png') for frame in v
                        ]
                        data_anno_list.append(record)
                cache_dir = osp.dirname(cache_file)
                if not osp.exists(cache_dir):
                    os.makedirs(cache_dir)
                with open(cache_file, 'wb') as f:
                    pickle.dump(data_anno_list, f)
                logger.info(
                    "Davis VOS dataset: cache dumped at: {}".format(cache_file))
                DavisDataset.data_items += data_anno_list
