from __future__ import absolute_import, print_function

import os
import glob
import six
import numpy as np


class TrackingNet(object):
    r"""`TrackingNet <https://tracking-net.org/>`_ Datasets.

    Publication:
        ``TrackingNet: A Large-Scale Dataset and Benchmark for Object Tracking in the Wild.``,
        M. Muller, A. Bibi, S. Giancola, S. Al-Subaihi and B. Ghanem, ECCV 2018.
    
    Args:
        root_dir (string): Root directory of dataset where sequence
            folders exist.
        subset (string, optional): Specify ``train`` or ``test``
            subset of TrackingNet.
    """
    def __init__(self, root_dir, subset='test'):
        super(TrackingNet, self).__init__()
        assert subset in ['train', 'test'], 'Unknown subset.'

        self.root_dir = root_dir
        self.subset = subset
        if subset == 'test':
            self.subset_dirs = ['TEST']
        elif subset == 'train':
            self.subset_dirs = ['TRAIN_%d' % c for c in range(12)]
        self._check_integrity(root_dir, self.subset_dirs)

        self.anno_files = [glob.glob(os.path.join(
            root_dir, c, 'anno/*.txt')) for c in self.subset_dirs]
        self.anno_files = sorted(sum(self.anno_files, []))
        self.seq_dirs = [os.path.join(
            os.path.dirname(os.path.dirname(f)),
            os.path.basename(f)[:-4])
            for f in self.anno_files]
        self.seq_names = [os.path.basename(d) for d in self.seq_dirs]

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
        
        img_files = glob.glob(
            os.path.join(self.seq_dirs[index], '*.jpg'))
        img_files = sorted(img_files, key=lambda x: int(x[:-4]))
        anno = np.loadtxt(self.anno_files[index], delimiter=',')
        assert len(img_files) == len(anno)
        assert anno.shape[1] == 4
        
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
