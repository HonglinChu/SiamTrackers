from __future__ import absolute_import, print_function, division

import os
import glob
import numpy as np
import six


class NfS(object):
    """`NfS <http://ci2cv.net/nfs/index.html>`_ Dataset.

    Publication:
        ``Need for Speed: A Benchmark for Higher Frame Rate Object Tracking``,
        H. K. Galoogahi, A. Fagg, C. Huang, D. Ramanan and S. Lucey, ICCV 2017.
    
    Args:
        root_dir (string): Root directory of dataset where sequence
            folders exist.
        fps (integer): Sequence frame rate. Two options ``30`` and ``240``
            are available. Default is 240.
    """
    def __init__(self, root_dir, fps=240):
        super(NfS, self).__init__()
        assert fps in [30, 240]
        self.fps = fps
        self.root_dir = root_dir
        self._check_integrity(root_dir)

        self.anno_files = sorted(glob.glob(
            os.path.join(root_dir, '*/%d/*.txt' % fps)))
        self.seq_names = [
            os.path.basename(f)[:-4] for f in self.anno_files]
        self.seq_dirs = [os.path.join(
            os.path.dirname(f), n)
            for f, n in zip(self.anno_files, self.seq_names)]
    
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
            os.path.join(self.seq_dirs[index], '*.jpg')))
        anno = np.loadtxt(self.anno_files[index], dtype=str)
        anno = anno[:, 1:5].astype(float)  # [left, top, right, bottom]
        anno[:, 2:] -= anno[:, :2]         # [left, top, width, height]

        # handle inconsistent lengths
        if not len(img_files) == len(anno):
            if abs(len(anno) / len(img_files) - 8) < 1:
                anno = anno[0::8, :]
            diff = abs(len(img_files) - len(anno))
            if diff > 0 and diff <= 1:
                n = min(len(img_files), len(anno))
                anno = anno[:n]
                img_files = img_files[:n]
        assert len(img_files) == len(anno)

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
