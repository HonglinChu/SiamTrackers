from __future__ import absolute_import, print_function

import os
import glob
import numpy as np
import pickle
import six

from loguru import logger
from tqdm import tqdm

_VALID_SUBSETS = ['train', 'val', 'test']


class GOT10k(object):
    r"""`GOT-10K <http://got-10k.aitestunion.com//>`_ Dataset.

    Publication:
        ``GOT-10k: A Large High-Diversity Benchmark for Generic Object
        Tracking in the Wild``, L. Huang, X. Zhao and K. Huang, ArXiv 2018.
    
    Args:
        root_dir (string): Root directory of dataset where ``train``,
            ``val`` and ``test`` folders exist.
        subset (string, optional): Specify ``train``, ``val`` or ``test``
            subset of GOT-10k.
        return_meta (string, optional): If True, returns ``meta``
            of each sequence in ``__getitem__`` function, otherwise
            only returns ``img_files`` and ``anno``.
        list_file (string, optional): If provided, only read sequences
            specified by the file instead of all sequences in the subset.
    """
    data_dict = {subset : dict() for subset in _VALID_SUBSETS}
    def __init__(self, root_dir, subset='test', return_meta=False,
                 list_file=None, check_integrity=True, cache_path=None, ignore_cache=False):
        super(GOT10k, self).__init__()
        assert subset in _VALID_SUBSETS, 'Unknown subset.'
        self.root_dir = root_dir
        self.subset = subset
        self.return_meta = False if subset == 'test' else return_meta
        self.cache_path = cache_path
        self.ignore_cache = ignore_cache

        if list_file is None:
            list_file = os.path.join(root_dir, subset, 'list.txt')
        if check_integrity:
            self._check_integrity(root_dir, subset, list_file)

        with open(list_file, 'r') as f:
            self.seq_names = f.read().strip().split('\n')
        # Former seq_dirs/anno_files have been replaced by caching mechanism.
        # See _ensure_cache for detail.
        # self.seq_dirs = [os.path.join(root_dir, subset, s)
        #                  for s in self.seq_names]
        # self.anno_files = [os.path.join(d, 'groundtruth.txt')
        #                    for d in self.seq_dirs]
        self._ensure_cache()

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
        if isinstance(index, int):
            seq_name = self.seq_names[index]
        else:
            if not index in self.seq_names:
                logger.error('Sequence {} not found.'.format(index))
                logger.error("Length of seq_names: %d"%len(self.seq_names))
                raise Exception('Sequence {} not found.'.format(index))
            seq_name = index
        img_files = GOT10k.data_dict[self.subset][seq_name]["img_files"]
        anno = GOT10k.data_dict[self.subset][seq_name]["anno"]

        if self.subset == 'test' and (anno.size // 4 == 1):
            anno = anno.reshape(-1, 4)
            # anno = anno[np.newaxis, :]
        else:
            assert len(img_files) == len(anno)

        if self.return_meta:
            meta = GOT10k.data_dict[self.subset][seq_name]["meta"]
            return img_files, anno, meta
        else:
            return img_files, anno

    def __len__(self):
        return len(self.seq_names)

    def _check_integrity(self, root_dir, subset, list_file=None):
        assert subset in ['train', 'val', 'test']
        if list_file is None:
            list_file = os.path.join(root_dir, subset, 'list.txt')

        if os.path.isfile(list_file):
            with open(list_file, 'r') as f:
                seq_names = f.read().strip().split('\n')

            # check each sequence folder
            for seq_name in seq_names:
                seq_dir = os.path.join(root_dir, subset, seq_name)
                if not os.path.isdir(seq_dir):
                    logger.error('Warning: sequence %s not exists.' % seq_name)
        else:
            # dataset not exists
            raise Exception('Dataset {} not found or corrupted.'.format(list_file))

    def _fetch_meta(self, seq_dir):
        # meta information
        meta_file = os.path.join(seq_dir, 'meta_info.ini')
        with open(meta_file) as f:
            meta = f.read().strip().split('\n')[1:]
        meta = [line.split(': ') for line in meta]
        meta = {line[0]: line[1] for line in meta}

        # attributes
        attributes = ['cover', 'absence', 'cut_by_image']
        for att in attributes:
            meta[att] = np.loadtxt(os.path.join(seq_dir, att + '.label'))

        return meta

    def _ensure_cache(self):
        """Perform all overheads related to cache (building/loading/check)
        """
        # check if subset cache already exists in GOT10k.data_dict and is valid w.r.t. list.txt
        if self._check_cache_for_current_subset():
            return

        # load subset cache into GOT10k.data_dict
        cache_path = self._get_cache_path(cache_path=self.cache_path)  #读取的 train.pkl 数据
        self.cache_path = cache_path
        if os.path.isfile(cache_path) and not self.ignore_cache:
            logger.info("{}: cache file exists: {} ".format(GOT10k.__name__, cache_path))
            self._load_cache_for_current_subset(cache_path)
            if self._check_cache_for_current_subset():
                logger.info("{}: record check has been processed and validity is confirmed for cache file: {} ".format(GOT10k.__name__, cache_path))
                return
            else:
                logger.info("{}: cache file {} not valid, rebuilding cache...".format(GOT10k.__name__, cache_path))
        # build subset cache in GOT10k.data_dict and cache to storage
        self._build_cache_for_current_subset()
        logger.info("{}: current cache file: {} ".format(GOT10k.__name__, self.cache_path))
        logger.info("{}: need to clean this cache file if you move dataset directory".format(GOT10k.__name__))
        logger.info("{}: consider cleaning this cache file in case of erros such as FileNotFoundError or IOError".format(GOT10k.__name__))

    def _check_cache_for_current_subset(self) -> bool:
        r""" check if GOT10k.data_dict[subset] exists and contains all record in seq_names
        """
        is_valid_data_dict = (self.subset in GOT10k.data_dict) and \
                             (set(GOT10k.data_dict[self.subset].keys()) == set(self.seq_names))
        return is_valid_data_dict

    def _get_cache_path(self, cache_path : str=None):
        r"""Ensure cache_path.
            If cache_path does not exist, turn to default set: root_dir/subset.pkl.
        """
        if (cache_path is None) or (not os.path.isfile(cache_path)):
            logger.info("{}: passed cache file {} invalid, change to default cache path".format(GOT10k.__name__, cache_path))
            cache_path = os.path.join(self.root_dir, self.subset+".pkl")
        return cache_path

    def _load_cache_for_current_subset(self, cache_path: str):
        assert os.path.exists(cache_path), "cache_path does not exist: %s "%cache_path
        with open(cache_path, "rb") as f:
            GOT10k.data_dict[self.subset] = pickle.load(f)
        logger.info("{}: loaded cache file {}".format(GOT10k.__name__, cache_path))

    def _build_cache_for_current_subset(self):
        r"""Build cache for current subset (self.subset)
        """
        root_dir = self.root_dir
        subset = self.subset
        logger.info("{}: start loading subset {}".format(GOT10k.__name__, subset))
        for seq_name in tqdm(self.seq_names):
            seq_dir = os.path.join(root_dir, subset, seq_name)
            img_files, anno, meta = self.load_single_sequence(seq_dir)
            GOT10k.data_dict[self.subset][seq_name] = dict(img_files = img_files, anno=anno, meta=meta)
        with open(self.cache_path, "wb") as f:
            pickle.dump(GOT10k.data_dict[self.subset], f)
        logger.info("{}: dump cache file to {}".format(GOT10k.__name__, self.cache_path))

    def load_single_sequence(self, seq_dir):
        img_files = sorted(glob.glob(os.path.join(
            seq_dir, '*.jpg')))
        anno = np.loadtxt(os.path.join(seq_dir, "groundtruth.txt"), delimiter=',')

        if self.subset == 'test' and anno.ndim == 1:
            assert len(anno) == 4
            anno = anno[np.newaxis, :]
        else:
            assert len(img_files) == len(anno)

        if self.return_meta or self.subset == "val":
            meta = self._fetch_meta(seq_dir)
            return img_files, anno, meta
        else:
            return img_files, anno, None
