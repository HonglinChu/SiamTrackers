from __future__ import absolute_import, print_function

import os
import glob
import json
import numpy as np
import six
from typing import Dict
import pickle
from tqdm import tqdm

from loguru import logger

_VALID_SUBSETS = ['train', 'test']


class LaSOT(object):
    r"""`LaSOT <https://cis.temple.edu/lasot/>`_ Datasets.

    Publication:
        ``LaSOT: A High-quality Benchmark for Large-scale Single Object Tracking``,
        H. Fan, L. Lin, F. Yang, P. Chu, G. Deng, S. Yu, H. Bai,
        Y. Xu, C. Liao, and H. Ling., CVPR 2019.
    
    Args:
        root_dir (string): Root directory of dataset where sequence
            folders exist.
        subset (string, optional): Specify ``train`` or ``test``
            subset of LaSOT.
    """
    data_dict = {subset : dict() for subset in _VALID_SUBSETS}
    def __init__(self, root_dir, subset='test', return_meta=False,
                 check_integrity=True, cache_path=None, ignore_cache=False):
        super(LaSOT, self).__init__()
        subset = subset.split('_')
        assert set(subset).issubset({'train', 'test'}), 'Unknown subset.'

        self.root_dir = root_dir
        self.subset = subset
        self.return_meta = return_meta
        # check seems useless, disabled 
        # if check_integrity:
        #     self._check_integrity(root_dir)
        self.cache_path = cache_path
        self.ignore_cache = ignore_cache

        self.anno_files = sorted(glob.glob(
            os.path.join(root_dir, '*/*/groundtruth.txt')))
        self.seq_dirs = [os.path.join(
            os.path.dirname(f), 'img') for f in self.anno_files]
        self.seq_names = [os.path.basename(
            os.path.dirname(f)) for f in self.anno_files]
        
        # load subset sequence names
        split_file = os.path.join(
            os.path.dirname(__file__), 'lasot.json')
        with open(split_file, 'r') as f:
            splits = json.load(f)
        self.splits = splits
        self.seq_names = []
        for s in subset:
            self.seq_names.extend(splits[s])

        # Former seq_dirs/anno_files have been replaced by caching mechanism.
        # See _ensure_cache for detail.
        # image and annotation paths
        # self.seq_dirs = [os.path.join(
        #     root_dir, n[:n.rfind('-')], n, 'img')
        #     for n in self.seq_names]
        # self.anno_files = [os.path.join(
        #     os.path.dirname(d), 'groundtruth.txt')
        #     for d in self.seq_dirs]
        self._ensure_cache()
        self.seq_names = [k for subset in self.subset
                            for k, _ in LaSOT.data_dict[subset].items()]
        self.seq_names = sorted(self.seq_names)
        self.seq_datas = {k : v for subset in self.subset
                               for k, v in LaSOT.data_dict[subset].items()}

    def __getitem__(self, index):
        r"""        
        Args:
            index (integer or string): Index or name of a sequence.
        
        Returns:
            tuple: (img_files, anno) if ``return_meta`` is False, otherwise
                (img_files, anno, meta), where ``img_files`` is a list of
                file names, ``anno`` is a N x 4 (rectangles) numpy array, while
                ``meta`` is a dict contains meta information about the sequence.
        """
        # if isinstance(index, six.string_types):
        #     if not index in self.seq_names:
        #         raise Exception('Sequence {} not found.'.format(index))
        #     index = self.seq_names.index(index)
        if isinstance(index, int):
            index = self.seq_names[index]
        
        seq_data = self.seq_datas[index]
        img_files = seq_data["img_files"]
        anno = seq_data["anno"]
        meta = seq_data["meta"]

        # img_files = sorted(glob.glob(os.path.join(
        #     self.seq_dirs[index], '*.jpg')))
        # anno = np.loadtxt(self.anno_files[index], delimiter=',')

        if self.return_meta:
            meta = self._fetch_meta(self.seq_dirs[index])
            return img_files, anno, meta
        else:
            return img_files, anno

    def __len__(self):
        return len(self.seq_names)

    def _check_integrity(self, root_dir):
        seq_names = os.listdir(root_dir)
        seq_names = [n for n in seq_names if not n[0] == '.']

        if os.path.isdir(root_dir) and len(seq_names) > 0:
            # check each sequence folder
            for seq_name in seq_names:
                seq_dir = os.path.join(root_dir, seq_name)
                if not os.path.isdir(seq_dir):
                    print('Warning: sequence %s not exists.' % seq_name)
        else:
            # dataset not exists
            raise Exception('Dataset not found or corrupted.')

    def _fetch_meta(self, seq_dir):
        seq_dir = os.path.dirname(seq_dir)
        meta = {}

        # attributes
        for att in ['full_occlusion', 'out_of_view']:
            att_file = os.path.join(seq_dir, att + '.txt')
            meta[att] = np.loadtxt(att_file, delimiter=',')
        
        # nlp
        nlp_file = os.path.join(seq_dir, 'nlp.txt')
        with open(nlp_file, 'r') as f:
            meta['nlp'] = f.read().strip()

        return meta

    def _ensure_cache(self):
        """Perform all overheads related to cache (building/loading/check)
        """
        # check if subset cache already exists in LaSOT.data_dict and is valid w.r.t. list.txt
        if self._check_cache_for_current_subset():
            return

        # load subset cache into LaSOT.data_dict
        cache_path = self._get_cache_path(cache_path=self.cache_path)
        self.cache_path = cache_path
        if all([os.path.isfile(p) for p in self.cache_path.values()]) and not self.ignore_cache:
            logger.info("{}: cache file exists: {} ".format(LaSOT.__name__, cache_path))
            self._load_cache_for_current_subset(cache_path)
            if self._check_cache_for_current_subset():
                logger.info("{}: record check has been processed and validity is confirmed for cache file: {} ".format(LaSOT.__name__, cache_path))
                return
            else:
                logger.info("{}: cache file {} not valid, rebuilding cache...".format(LaSOT.__name__, cache_path))
        # build subset cache in LaSOT.data_dict and cache to storage
        self._build_cache_for_current_subset()
        logger.info("{}: current cache file: {} ".format(LaSOT.__name__, self.cache_path))
        logger.info("{}: need to clean this cache file if you move dataset directory".format(LaSOT.__name__))
        logger.info("{}: consider cleaning this cache file in case of erros such as FileNotFoundError or IOError".format(LaSOT.__name__))


    def _get_cache_path(self, cache_path : Dict[str, str]=None):
        r"""Ensure cache_path.
            If cache_path does not exist, turn to default set: root_dir/subset.pkl.
        """
        if (cache_path is None) or any([not os.path.isfile(cache_path) for p in cache_path.values()]):
            logger.info("{}: passed cache file {} invalid, change to default cache path".format(LaSOT.__name__, cache_path))
            cache_path = {subset : os.path.join(self.root_dir, subset+".pkl") for subset in self.subset}
        return cache_path

    def _check_cache_for_current_subset(self) -> bool:
        r""" check if LaSOT.data_dict[subset] exists and contains all record in seq_names
        """
        is_valid_data_dict = all([subset in LaSOT.data_dict for subset in self.subset]) and \
                             (set([seq_name for subset in self.subset for seq_name in LaSOT.data_dict[subset].keys()]) == set(self.seq_names))
        return is_valid_data_dict

    def _build_cache_for_current_subset(self):
        r"""Build cache for current subset (self.subset)
        """
        root_dir = self.root_dir
        subset = self.subset
        for s in subset:
            logger.info("{}: start loading {}".format(LaSOT.__name__, s))
            seq_names = self.splits[s]
            for seq_name in tqdm(seq_names):
                seq_dir = os.path.join(root_dir, seq_name[:seq_name.rfind('-')], seq_name)
                img_files, anno, meta = self.load_single_sequence(seq_dir)
                LaSOT.data_dict[s][seq_name] = dict(img_files = img_files, anno=anno, meta=meta)
            with open(self.cache_path[s], "wb") as f:
                pickle.dump(LaSOT.data_dict[s], f)
            logger.info("{}: dump cache file to {}".format(LaSOT.__name__, self.cache_path[s]))

    def _load_cache_for_current_subset(self, cache_path: Dict[str, str]):
        for subset in self.subset:
            assert os.path.exists(cache_path[subset]), "cache_path does not exist: %s "%cache_path[subset]
            with open(cache_path[subset], "rb") as f:
                LaSOT.data_dict[subset] = pickle.load(f)
            logger.info("{}: loaded cache file {}".format(LaSOT.__name__, cache_path[subset]))

    def load_single_sequence(self, seq_dir):
        img_files = sorted(glob.glob(os.path.join(
            seq_dir, 'img/*.jpg')))
        anno = np.loadtxt(os.path.join(seq_dir, "groundtruth.txt"), delimiter=',')

        assert len(img_files) == len(anno)

        if self.return_meta:
            meta = self._fetch_meta(seq_dir)
            return img_files, anno, meta
        else:
            return img_files, anno, None
