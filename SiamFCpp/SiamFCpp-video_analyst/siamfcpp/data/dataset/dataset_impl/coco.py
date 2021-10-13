# -*- coding: utf-8 -*-
import contextlib
import io
import os
import os.path as osp
import pickle

import cv2
import numpy as np
from loguru import logger
from pycocotools import mask as MaskApi
from pycocotools.coco import COCO

from siamfcpp.data.dataset.dataset_base import (TRACK_DATASETS,
                                                    VOS_DATASETS, DatasetBase)
from siamfcpp.pipeline.utils.bbox import xywh2xyxy


@TRACK_DATASETS.register
@VOS_DATASETS.register
class COCODataset(DatasetBase):
    r"""
    COCO dataset helper
    Hyper-parameters
    ----------------
    dataset_root: str
        path to root of the dataset
    subsets: list
        dataset split name [train2017,val2017]
    ratio: float
        dataset ratio. used by sampler (data.sampler).
    """
    data_items = []
    _DUMMY_ANNO = [[-1, -1, 0, 0]]

    default_hyper_params = dict(
        dataset_root="datasets/coco2017",
        subsets=[
            "val2017",
        ],
        ratio=1.0,
        with_mask=False,
    )

    def __init__(self) -> None:
        r"""
        Create dataset with config
        """
        super(COCODataset, self).__init__()
        self._state["dataset"] = None

    def update_params(self):
        r"""
        an interface for update params
        """
        dataset_root = self._hyper_params["dataset_root"]
        self._hyper_params["dataset_root"] = osp.realpath(dataset_root)
        if len(COCODataset.data_items) == 0:
            self._ensure_cache()

    def _generate_mask_from_anno(self, raw_mask, img_h, img_w):
        jth_mask_raw = MaskApi.frPyObjects(raw_mask, img_h, img_w)
        jth_mask = MaskApi.decode(jth_mask_raw)
        mask_shape = jth_mask.shape
        if len(mask_shape) == 3:
            target_mask = np.zeros((mask_shape[0], mask_shape[1]),
                                   dtype=np.uint8)
            for iter_chl in range(mask_shape[2]):
                target_mask = target_mask | jth_mask[:, :, iter_chl]
        else:
            target_mask = jth_mask
        target_mask = target_mask.astype(np.uint8)  # 全部是0或者1
        return target_mask

    def __getitem__(self, item):
        """
        :param item: int, video id
        :return:
            image_files
            annos
            meta (optional)
        """
        record = COCODataset.data_items[item]
        image_file = record["file_name"]
        img_h = record["height"]
        img_w = record["width"]
        anno = record['annotations']
        if self._hyper_params["with_mask"]:
            mask_anno = []
            for obj in anno:
                raw_mask = obj['segmentation']
                mask = self._generate_mask_from_anno(raw_mask, img_h, img_w)
                mask_anno.append(mask)

            sequence_data = dict(image=[image_file], anno=mask_anno)
        else:
            box_anno = []
            for obj in anno:
                box_anno.append(obj['bbox'])
            if len(box_anno) <= 0:
                box_anno = self._DUMMY_ANNO
            box_anno = xywh2xyxy(box_anno)
            sequence_data = dict(image=[image_file], anno=box_anno)

        return sequence_data

    def __len__(self):
        return len(COCODataset.data_items)

    def _ensure_cache(self):
        dataset_root = self._hyper_params["dataset_root"]
        subsets = self._hyper_params["subsets"]
        for subset in subsets:
            data_anno_list = []
            image_root = osp.join(dataset_root, subset)
            if self._hyper_params["with_mask"]:
                cache_file = osp.join(dataset_root,
                                      "cache/coco_mask_{}.pkl".format(subset))
            else:
                cache_file = osp.join(dataset_root,
                                      "cache/coco_bbox_{}.pkl".format(subset))
            if osp.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    COCODataset.data_items += pickle.load(f)
                logger.info("{}: loaded cache file {}".format(
                    COCODataset.__name__, cache_file))
            else:
                anno_file = osp.join(
                    dataset_root,
                    "annotations/instances_{}.json".format(subset))
                with contextlib.redirect_stdout(io.StringIO()):
                    coco_api = COCO(anno_file)
                    # sort indices for reproducible results
                    img_ids = sorted(coco_api.imgs.keys())
                    # imgs is a list of dicts, each looks something like:
                    # {'license': 4,
                    #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
                    #  'file_name': 'COCO_val2014_000000001268.jpg',
                    #  'height': 427,
                    #  'width': 640,
                    #  'date_captured': '2013-11-17 05:57:24',
                    #  'id': 1268}
                    imgs = coco_api.loadImgs(img_ids)
                    # anns is a list[list[dict]], where each dict is an annotation
                    # record for an object. The inner list enumerates the objects in an image
                    # and the outer list enumerates over images. Example of anns[0]:
                    # [{'segmentation': [[192.81,
                    #     247.09,
                    #     ...
                    #     219.03,
                    #     249.06]],
                    #   'area': 1035.749,
                    #   'iscrowd': 0,
                    #   'image_id': 1268,
                    #   'bbox': [192.81, 224.8, 74.73, 33.43],
                    #   'category_id': 16,
                    #   'id': 42986},
                    #  ...]
                    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]

                if "minival" not in anno_file:
                    # The popular valminusminival & minival annotations for COCO2014 contain this bug.
                    # However the ratio of buggy annotations there is tiny and does not affect accuracy.
                    # Therefore we explicitly white-list them.
                    ann_ids = [
                        ann["id"] for anns_per_image in anns
                        for ann in anns_per_image
                    ]
                    assert len(set(ann_ids)) == len(
                        ann_ids
                    ), "Annotation ids in '{}' are not unique!".format(
                        anno_file)

                imgs_anns = list(zip(imgs, anns))
                ann_keys = ["iscrowd", "bbox", "keypoints", "category_id"]
                # iterate over annotation
                for (img_dict, anno_dict_list) in imgs_anns:
                    record = {}
                    record["file_name"] = os.path.join(image_root,
                                                       img_dict["file_name"])
                    record["height"] = img_dict["height"]
                    record["width"] = img_dict["width"]
                    image_id = record["image_id"] = img_dict["id"]

                    objs = []
                    for anno in anno_dict_list:
                        # Check that the image_id in this annotation is the same as
                        # the image_id we're looking at.
                        # This fails only when the data parsing logic or the annotation file is buggy.

                        # The original COCO valminusminival2014 & minival2014 annotation files
                        # actually contains bugs that, together with certain ways of using COCO API,
                        # can trigger this assertion.
                        assert anno["image_id"] == image_id, logger.error(
                            "{} vs {}".format(anno["image_id"], image_id))

                        assert anno.get(
                            "ignore", 0
                        ) == 0, '"ignore" in COCO json file is not supported.'

                        obj = {
                            key: anno[key]
                            for key in ann_keys if key in anno
                        }

                        segm = anno.get("segmentation", None)
                        if segm:  # either list[list[float]] or dict(RLE)
                            if not isinstance(segm, dict):
                                # filter out invalid polygons (< 3 points)
                                segm = [
                                    poly for poly in segm
                                    if len(poly) % 2 == 0 and len(poly) >= 6
                                ]
                                if len(segm) == 0:
                                    num_instances_without_valid_segmentation += 1
                                    continue  # ignore this instance
                            obj["segmentation"] = segm
                        else:
                            if self._hyper_params["with_mask"]:
                                continue
                        objs.append(obj)
                    # filter out image without any targets
                    if len(objs) == 0:
                        continue
                    record["annotations"] = objs
                    data_anno_list.append(record)

                # save internal .json file
                cache_dir = osp.dirname(cache_file)
                if not osp.exists(cache_dir):
                    os.makedirs(cache_dir)
                with open(cache_file, 'wb') as f:
                    pickle.dump(data_anno_list, f)
                logger.info(
                    "COCO dataset: cache dumped at: {}".format(cache_file))
                COCODataset.data_items += data_anno_list
