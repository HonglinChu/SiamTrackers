from __future__ import absolute_import, print_function

import os
import glob
import six
import numpy as np

from typing import List, Tuple, Dict
import pickle
from loguru import logger
from tqdm import tqdm

_VALID_SUBSETS = ["TRAIN_%d"%i for i in range(12)] + ["TEST"]

class TrackingNet(object):
    r"""`TrackingNet <https://tracking-net.org/>`_ Datasets.

    Publication:
        ``TrackingNet: A Large-Scale Dataset and Benchmark for Object Tracking in the Wild.``,
        M. Muller, A. Bibi, S. Giancola, S. Al-Subaihi and B. Ghanem, ECCV 2018.
    
    Args:
        root_dir (string): Root directory of dataset where sequence
            folders exist.
        subset (string, optional): Specify ``train`` or ``test`` or ``train0,1,...``
            subset of TrackingNet.
    """
    data_dict = {subset : dict() for subset in _VALID_SUBSETS}
    def __init__(self, root_dir, subset='test', *args, 
                 check_integrity=True, cache_dir=None, ignore_cache=False, **kwargs):
        super(TrackingNet, self).__init__()
        assert subset.startswith(('train', 'test')), 'Unknown subset.'

        self.root_dir = root_dir
        self.subset = subset
        if subset == 'test':
            self.subset_dirs = ['TEST']
        elif subset == 'train':
            self.subset_dirs = ['TRAIN_%d' % c for c in range(12)]
        else:
            subset = subset[len('train'):]
            chunk_ids = [int(s) for s in subset.split(",")]
            self.subset_dirs = ['TRAIN_%d' % c for c in chunk_ids]
        self.cache_dir = cache_dir
        self.ignore_cache = ignore_cache
        self._check_integrity(root_dir, self.subset_dirs)

        # self.anno_files = [glob.glob(os.path.join(
        #     root_dir, c, 'anno/*.txt')) for c in self.subset_dirs]
        # self.anno_files = sorted(sum(self.anno_files, []))
        # self.seq_dirs = [os.path.join(
        #     os.path.dirname(os.path.dirname(f)),
        #     'frames',
        #     os.path.basename(f)[:-4])
        #     for f in self.anno_files]
        # self.seq_names = [os.path.basename(d) for d in self.seq_dirs]

        self._ensure_cache()
        # fusion of subsets
        self.seq_names = [k for subset in self.subset_dirs
                            for k, _ in TrackingNet.data_dict[subset].items()]
        self.seq_names = sorted(self.seq_names)
        self.seq_datas = {k : v for subset in self.subset_dirs
                               for k, v in TrackingNet.data_dict[subset].items()}

    def __getitem__(self, index):
        r"""        
        Args:
            index (integer or string): Index or name of a sequence.
        
        Returns:
            tuple: (img_files, anno), where ``img_files`` is a list of
                file names and ``anno`` is a N x 4 (rectangles) numpy array.
        """
        # if isinstance(index, six.string_types):
        #     if not index in self.seq_names:
        #         raise Exception('Sequence {} not found.'.format(index))
        #     index = self.seq_names.index(index)
        
        # img_files = glob.glob(
        #     os.path.join(self.seq_dirs[index], '*.jpg'))
        # img_files = sorted(img_files, key=lambda x: int(os.path.basename(x)[:-4]))
        # anno = np.loadtxt(self.anno_files[index], delimiter=',')

        # if self.subset.startswith('train'):
        #     assert len(img_files) == len(anno)
        #     assert anno.shape[1] == 4
        # elif self.subset=='test':
        #     assert anno.shape[0] == 4
        # anno = anno.reshape(-1, 4)
        if isinstance(index, six.string_types):
            seq_data = self.seq_datas[index]
        else:
            seq_name = self.seq_names[index]
            seq_data = self.seq_datas[seq_name]

        img_files = seq_data["img_files"]
        anno = seq_data["anno"]
        
        return img_files, anno

    def __len__(self):
        return len(self.seq_names)

    def _check_integrity(self, root_dir, subset_dirs):
        # check each subset path
        for c in subset_dirs:
            subset_dir = os.path.join(root_dir, c)
            
            # check data and annotation folders
            for folder in ['anno', 'frames']:
                if not os.path.isdir(os.path.join(subset_dir, folder)):
                    raise Exception('Dataset not found or corrupted.')

    def _ensure_cache(self):
        # load subset cache into TrackingNet.data_dict
        self.cache_path_dict = self._get_cache_path_dict(cache_dir=self.cache_dir)

        for subset in self.subset_dirs:
            cache_path = self.cache_path_dict[subset]
            # check if subset cache already exists in TrackingNet.data_dict and is valid w.r.t. list.txt
            if self._check_cache_for_specific_subset(subset):
                logger.info("{}: record check has been processed and validity is confirmed for cache file: {} ".format(TrackingNet.__name__, cache_path))
                continue
            if os.path.isfile(cache_path) and not self.ignore_cache:
                logger.info("{}: cache file exists: {} ".format(TrackingNet.__name__, cache_path))
                self._load_cache_for_specific_subset(cache_path, subset)
                if self._check_cache_for_specific_subset(subset):
                    logger.info("{}: record check has been processed and validity is confirmed for cache file: {} ".format(TrackingNet.__name__, cache_path))
                    continue
                else:
                    logger.info("{}: cache file {} not valid, rebuilding cache...".format(TrackingNet.__name__, cache_path))
            self._build_cache_for_specific_subset(subset)
            logger.info("{}: cache file built at: {}".format(TrackingNet.__name__, cache_path))

    def _check_cache_for_specific_subset(self, subset) -> bool:
        r""" check if TrackingNet.data_dict[subset] exists and contains all record in seq_names
        """
        is_subset_valid = all([ (subset in TrackingNet.data_dict) for subset in self.subset_dirs ])
        cached_seq_names = [seq_name
            for seq_name in TrackingNet.data_dict[subset] 
        ]
        seq_names = self._get_seq_names_for_specific_subset(subset)
        is_seq_names_valid = (set(seq_names) == set(cached_seq_names))

        return (is_subset_valid and is_seq_names_valid)

    def _get_cache_path_dict(self, cache_dir : str=None):
        r"""Ensure cache_path.
            If cache_path does not exist, turn to default set: root_dir/subset.pkl.
        """
        if (cache_dir is None) or (not os.path.exists(cache_dir)):
            logger.info("{}: passed cache dir {} invalid, change to default cache dir".format(TrackingNet.__name__, cache_dir))
            cache_dir = os.path.join(self.root_dir)# self.subset+".pkl")

        cache_path_dict = {subset : os.path.join(cache_dir, "%s.pkl"%subset) for subset in _VALID_SUBSETS}

        return cache_path_dict

    def _load_cache_for_specific_subset(self, cache_path: str, subset: str):
        assert os.path.exists(cache_path), "cache_path does not exist: %s "%cache_path
        with open(cache_path, "rb") as f:
            TrackingNet.data_dict[subset] = pickle.load(f)
        # logger.info("{}: loaded cache file {}".format(TrackingNet.__name__, cache_path))

    def _build_cache_for_specific_subset(self, subset: str):
        r"""Build cache for specific subset
        """
        # root_dir = self.root_dir
        logger.info("{}: start loading subset {}".format(TrackingNet.__name__, subset))
        seq_names = self._get_seq_names_for_specific_subset(subset)
        cache_path = self.cache_path_dict[subset]
        for seq_name in tqdm(seq_names):
            # seq_dir = os.path.join(root_dir, subset, seq_name)
            img_files, anno = self.load_single_sequence(subset, seq_name)
            TrackingNet.data_dict[subset][seq_name] = dict(img_files = img_files, anno=anno)
        with open(cache_path, "wb") as f:
            pickle.dump(TrackingNet.data_dict[subset], f)
        logger.info("{}: dump cache file to {}".format(TrackingNet.__name__, cache_path))
    
    def _get_seq_names_for_specific_subset(self, subset) -> List[str]:
        subset_dir = os.path.join(self.root_dir, subset)
        anno_file_pattern = os.path.join(subset_dir, 'anno/*.txt')
        anno_files = glob.glob(anno_file_pattern)
        seq_names = [os.path.basename(f)[:-len(".txt")] for f in anno_files]
        return seq_names

    def load_single_sequence(self, subset: str, seq_name: str) -> Tuple[List[str], np.array]:
        # img_files = sorted(glob.glob(os.path.join(
        #     seq_dir, '*.jpg')))
        # anno = np.loadtxt(os.path.join(seq_dir, "groundtruth.txt"), delimiter=',')

        img_file_pattern = os.path.join(self.root_dir, subset, "frames", seq_name, "*.jpg")

        img_files = glob.glob(img_file_pattern)
        img_files = sorted(img_files, key=lambda x: int(os.path.basename(x)[:-4]))
        anno_file = os.path.join(self.root_dir, subset, "anno/%s.txt"%seq_name)
        anno = np.loadtxt(anno_file, delimiter=',')

        if subset == 'TEST' and anno.ndim == 1:
            assert len(anno) == 4
            anno = anno[np.newaxis, :]
        else:
            assert len(img_files) == len(anno)

        return img_files, anno
