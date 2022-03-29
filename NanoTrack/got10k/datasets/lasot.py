# coding=UTF-8

from __future__ import absolute_import, print_function

import os
import glob
import json
import numpy as np
import six

class LaSOT(object):
    """`LaSOT <https://cis.temple.edu/lasot/>`_ Datasets.

    Publication:
    
        ``LaSOT: A High-quality Benchmark for Large-scale Single Object Tracking``,

        H. Fan, L. Lin, F. Yang, P. Chu, G. Deng, S. Yu, H. Bai

        Y. Xu, C. Liao, and H. Ling., CVPR 2019.
    
    Args:
        root_dir (string): Root directory of dataset where sequence
            folders exist.
        subset (string, optional): Specify ``train`` or ``test``
            subset of LaSOT.
    """
    def __init__(self, root_dir, subset='train', return_meta=False):

        super(LaSOT, self).__init__()

        assert subset in ['train', 'test'], 'Unknown subset.'

        self.root_dir = root_dir

        self.subset = subset

        self.return_meta = return_meta

        self._check_integrity(root_dir)

        self.anno_files = sorted(glob.glob(os.path.join(root_dir, '*/*/groundtruth.txt')))

        self.seq_dirs = [os.path.join(os.path.dirname(f), 'img') for f in self.anno_files]

        #self.seq_names = [os.path.basename(os.path.dirname(f) for f in self.anno_files)]
        
        # load subset sequence names
        split_file = os.path.join(os.path.dirname(__file__), 'lasot.json')   #__file__ 代表当前文件所在的路径全名，包括当前文件名，而 dirname
        #os.path.dirname 为获取当前文件所在的路径名，不包括当前文件名  os.path.basename 返回path最后的文件名
        with open(split_file, 'r') as f:

            splits = json.load(f)

        self.seq_names = splits[subset]

        # image and annotation paths
        self.seq_dirs = [os.path.join(root_dir, n[:n.rfind('-')], n, 'img')

            for n in self.seq_names]

        self.anno_files = [os.path.join(os.path.dirname(d), 'groundtruth.txt')

            for d in self.seq_dirs]

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
        if isinstance(index, six.string_types):
            if not index in self.seq_names:
                raise Exception('Sequence {} not found.'.format(index))
            index = self.seq_names.index(index)

        img_files = sorted(glob.glob(os.path.join(
            self.seq_dirs[index], '*.jpg')))
        anno = np.loadtxt(self.anno_files[index], delimiter=',')

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
