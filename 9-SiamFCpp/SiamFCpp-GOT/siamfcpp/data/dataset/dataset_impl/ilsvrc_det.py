import glob
import os
import os.path as osp
import pickle
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from tqdm import tqdm
from yacs.config import CfgNode

from siamfcpp.data.dataset.dataset_base import TRACK_DATASETS, DatasetBase
from siamfcpp.evaluation.got_benchmark.datasets import ImageNetVID
from siamfcpp.pipeline.utils.bbox import xywh2xyxy

# from .utils import Dataset, extract_fname, LazzyList

_VALID_SUBSETS = ['train', 'val']


@TRACK_DATASETS.register
class DETDataset(DatasetBase):
    r"""
    ILSVRC2015-DET dataset helper

    Hyper-parameters
    ----------------
    dataset_root: str
        path to root of the dataset
    subset: str
        dataset split name (train|val)
    ratio: float
        dataset ratio. used by sampler (data.sampler).
    max_diff: int
        maximum difference in index of a pair of sampled frames 
    """
    data_dict = {subset: dict() for subset in _VALID_SUBSETS}
    _DUMMY_ANNO = [[-1, -1, 0, 0]]

    # data_dirname = "Data"
    # anno_dirname = "Annotations"
    # image_path = dict(train="DET/train/*/*/*.xml", val="DET/val/*.xml")

    default_hyper_params = dict(
        dataset_root="datasets/ILSVRC2015",
        subset="train",
        ratio=1.0,
    )

    def __init__(self) -> None:
        super(DETDataset, self).__init__()
        self._state["dataset"] = None

    def update_params(self):
        r"""
        an interface for update params
        """
        dataset_root = self._hyper_params["dataset_root"]
        subset = self._hyper_params["subset"]
        self._hyper_params["dataset_root"] = osp.realpath(dataset_root)
        self._ensure_cache()
        self.im_names = list(DETDataset.data_dict[subset].keys())

    def __getitem__(self, item):
        """

        :param item: int, video id
        :return:
            image_files
            annos
            meta (optional)
        """
        subset = self._hyper_params["subset"]
        im_name = self.im_names[item]

        image_file = DETDataset.data_dict[subset][im_name]["image_file"]
        anno = DETDataset.data_dict[subset][im_name]["anno"]
        if len(anno) <= 0:
            anno = self._DUMMY_ANNO
        anno = xywh2xyxy(anno)

        sequence_data = dict(image=[image_file], anno=anno)

        return sequence_data

    def __len__(self):
        return len(self.im_names)

    def _ensure_cache(self):
        dataset_root = self._hyper_params["dataset_root"]
        subset = self._hyper_params["subset"]
        cache_dir = osp.join(dataset_root, "cache/det")
        cache_file = osp.join(cache_dir, "%s.pkl" % subset)
        # dataset_name = type(self).__name__

        if osp.exists(cache_file):
            # print("%s: using meta data file under ./meta_data" % dataset_name)
            with open(cache_file, 'rb') as f:
                DETDataset.data_dict[subset] = pickle.load(f)
        else:
            data_dirname = "Data"
            anno_dirname = "Annotations"
            data_path = dict(train="DET/train/*/*/*.JPEG", val="DET/val/*.JPEG")
            anno_path = dict(train="DET/train/*/*/*.xml", val="DET/val/*.xml")

            anno_dir = osp.join(dataset_root, anno_dirname)
            data_dir = osp.join(dataset_root, data_dirname)

            # print("%s: using external meta data file & generate private meta data under ./meta_data" % dataset_name)
            anno_file_pattern = osp.join(anno_dir, anno_path[subset])
            anno_files = sorted(glob.glob(anno_file_pattern))

            data_file_pattern = osp.join(data_dir, data_path[subset])
            data_files = sorted(glob.glob(data_file_pattern))

            # check integrity
            assert len(anno_files) == len(data_files)
            assert set([
                osp.splitext(osp.basename(p))[0] for p in anno_files
            ]) == set([osp.splitext(osp.basename(p))[0] for p in data_files])

            for data_file, anno_file in tqdm(list(zip(data_files, anno_files))):
                im_name = osp.splitext(osp.basename(data_file))[0]
                assert im_name == osp.splitext(osp.basename(anno_file))[0]
                anno = self._decode_det_anno(anno_file)
                DETDataset.data_dict[subset][im_name] = dict(
                    image_file=data_file,
                    anno=anno,
                )

            # dump cache
            if not osp.exists(cache_dir):
                os.makedirs(cache_dir)
            with open(cache_file, 'wb') as f:
                pickle.dump(DETDataset.data_dict[subset], f)

    def _decode_det_anno(self, p):
        tree = ET.parse(p)
        root = tree.getroot()

        anno = list()
        for obj in root.findall("object"):
            # trackid = int(obj.find("trackid").text)
            bbox = [
                float(obj.find("bndbox/xmin").text),
                float(obj.find("bndbox/ymin").text),
                float(obj.find("bndbox/xmax").text),
                float(obj.find("bndbox/ymax").text),
            ]
            rect = [
                bbox[0], bbox[1], bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1
            ]
            anno.append(rect)
        return anno
