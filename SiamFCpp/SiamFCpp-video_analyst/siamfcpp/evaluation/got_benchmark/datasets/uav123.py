from __future__ import absolute_import, print_function

import os
import glob
import numpy as np
import six
import json


class UAV123(object):
    """`UAV123 <https://ivul.kaust.edu.sa/Pages/pub-benchmark-simulator-uav.aspx>`_ Dataset.

    Publication:
        ``A Benchmark and Simulator for UAV Tracking``,
        M. Mueller, N. Smith and B. Ghanem, ECCV 2016.
    
    Args:
        root_dir (string): Root directory of dataset where sequence
            folders exist.
        version (integer or string): Specify the benchmark version, specify as one of
            ``UAV123`` and ``UAV20L``.
    """
    def __init__(self, root_dir, version='UAV123'):
        super(UAV123, self).__init__()
        assert version.upper() in ['UAV20L', 'UAV123']

        self.root_dir = root_dir
        self.version = version.upper()
        self._check_integrity(root_dir, version)

        # sequence meta information
        meta_file = os.path.join(
            os.path.dirname(__file__), 'uav123.json')
        with open(meta_file) as f:
            self.seq_metas = json.load(f)

        # sequence and annotation paths
        self.anno_files = sorted(glob.glob(
            os.path.join(root_dir, 'anno/%s/*.txt' % version)))
        self.seq_names = [
            os.path.basename(f)[:-4] for f in self.anno_files]
        self.seq_dirs = [os.path.join(
            root_dir, 'data_seq/UAV123/%s' % \
                self.seq_metas[version][n]['folder_name'])
            for n in self.seq_names]
    
    def __getitem__(self, index):
        r"""        
        Args:
            index (integer or string): Index or name of a sequence.
        
        Returns:
            tuple: (img_files, anno), where ``img_files`` is a list of
                file names and ``anno`` is a N x 4 (rectangles) numpy array.
        """
        if isinstance(index, six.string_types):
            if not index in self.seq_names:
                raise Exception('Sequence {} not found.'.format(index))
            index = self.seq_names.index(index)

        # valid frame range
        start_frame = self.seq_metas[self.version][
            self.seq_names[index]]['start_frame']
        end_frame = self.seq_metas[self.version][
            self.seq_names[index]]['end_frame']
        img_files = [os.path.join(
            self.seq_dirs[index], '%06d.jpg' % f)
            for f in range(start_frame, end_frame + 1)]

        # load annotations
        anno = np.loadtxt(self.anno_files[index], delimiter=',')
        assert len(img_files) == len(anno)
        assert anno.shape[1] == 4

        return img_files, anno

    def __len__(self):
        return len(self.seq_names)

    def _check_integrity(self, root_dir, version):
        # sequence meta information
        meta_file = os.path.join(
            os.path.dirname(__file__), 'uav123.json')
        with open(meta_file) as f:
            seq_metas = json.load(f)
        seq_names = list(seq_metas[version].keys())

        if os.path.isdir(root_dir) and len(os.listdir(root_dir)) > 3:
            # check each sequence folder
            for seq_name in seq_names:
                seq_dir = os.path.join(
                    root_dir, 'data_seq/UAV123/%s' % \
                        seq_metas[version][seq_name]['folder_name'])
                if not os.path.isdir(seq_dir):
                    print('Warning: sequence %s not exists.' % seq_name)
        else:
            # dataset not exists
            raise Exception('Dataset not found or corrupted.')
