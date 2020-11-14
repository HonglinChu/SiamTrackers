from __future__ import absolute_import, print_function

import os
import glob
import numpy as np
import six

from ..utils.ioutils import download, extract


class TColor128(object):
    """`TColor128 <http://www.dabi.temple.edu/~hbling/data/TColor-128/TColor-128.html>`_ Dataset.

    Publication:
        ``Encoding color information for visual tracking: algorithms and benchmark``,
        P. Liang, E. Blasch and H. Ling, TIP, 2015.
    
    Args:
        root_dir (string): Root directory of dataset where sequence
            folders exist.
    """
    def __init__(self, root_dir, download=True):
        super(TColor128, self).__init__()
        self.root_dir = root_dir
        if download:
            self._download(root_dir)
        self._check_integrity(root_dir)

        self.anno_files = sorted(glob.glob(
            os.path.join(root_dir, '*/*_gt.txt')))
        self.seq_dirs = [os.path.dirname(f) for f in self.anno_files]
        self.seq_names = [os.path.basename(d) for d in self.seq_dirs]
        # valid frame range for each sequence
        self.range_files = [glob.glob(
            os.path.join(d, '*_frames.txt'))[0]
            for d in self.seq_dirs]
    
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

        # load valid frame range
        frames = np.loadtxt(
            self.range_files[index], dtype=int, delimiter=',')
        img_files = [os.path.join(
            self.seq_dirs[index], 'img/%04d.jpg' % f)
            for f in range(frames[0], frames[1] + 1)]

        # load annotations
        anno = np.loadtxt(self.anno_files[index], delimiter=',')
        assert len(img_files) == len(anno)
        assert anno.shape[1] == 4

        return img_files, anno

    def __len__(self):
        return len(self.seq_names)

    def _download(self, root_dir):
        if not os.path.isdir(root_dir):
            os.makedirs(root_dir)
        elif len(os.listdir(root_dir)) > 100:
            print('Files already downloaded.')
            return

        url = 'http://www.dabi.temple.edu/~hbling/data/TColor-128/Temple-color-128.zip'
        zip_file = os.path.join(root_dir, 'Temple-color-128.zip')
        print('Downloading to %s...' % zip_file)
        download(url, zip_file)
        print('\nExtracting to %s...' % root_dir)
        extract(zip_file, root_dir)

        return root_dir

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
            raise Exception('Dataset not found or corrupted. ' +
                            'You can use download=True to download it.')
