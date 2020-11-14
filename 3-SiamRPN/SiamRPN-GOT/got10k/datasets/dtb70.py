from __future__ import absolute_import, print_function

import os
import glob
import numpy as np
import six


class DTB70(object):
    """`DTB70 <https://github.com/flyers/drone-tracking>`_ Dataset.

    Publication:
        ``Visual object tracking for unmanned aerial vehicles: A benchmark and new motion models``,
        Y. Wu, J. Lim and M.-H. Yang, IEEE TPAMI 2015.
    
    Args:
        root_dir (string): Root directory of dataset where sequence
            folders exist.
    """
    def __init__(self, root_dir):
        super(DTB70, self).__init__()
        self.root_dir = root_dir
        self._check_integrity(root_dir)

        self.anno_files = sorted(glob.glob(
            os.path.join(root_dir, '*/groundtruth_rect.txt')))
        self.seq_dirs = [os.path.dirname(f) for f in self.anno_files]
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

        img_files = sorted(glob.glob(
            os.path.join(self.seq_dirs[index], 'img/*.jpg')))
        anno = np.loadtxt(self.anno_files[index], delimiter=',')
        assert len(img_files) == len(anno)
        assert anno.shape[1] == 4

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
