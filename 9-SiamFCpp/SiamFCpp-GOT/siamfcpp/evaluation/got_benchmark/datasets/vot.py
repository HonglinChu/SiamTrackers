from __future__ import absolute_import, print_function, division

import os
import glob
import numpy as np
import six
import json
import hashlib

from ..utils.ioutils import download, extract


class VOT(object):
    r"""`VOT <http://www.votchallenge.net/>`_ Datasets.

    Publication:
        ``The Visual Object Tracking VOT2017 challenge results``, M. Kristan, A. Leonardis
            and J. Matas, etc. 2017.
    
    Args:
        root_dir (string): Root directory of dataset where sequence
            folders exist.
        version (integer, optional): Specify the benchmark version. Specify as
            one of 2013~2018. Default is 2017.
        anno_type (string, optional): Returned annotation types, chosen as one of
            ``rect`` and ``corner``. Default is ``rect``.
        download (boolean, optional): If True, downloads the dataset from the internet
            and puts it in root directory. If dataset is downloaded, it is not
            downloaded again.
        return_meta (string, optional): If True, returns ``meta``
            of each sequence in ``__getitem__`` function, otherwise
            only returns ``img_files`` and ``anno``.
        list_file (string, optional): If provided, only read sequences
            specified by the file.
    """
    __valid_versions = [2013, 2014, 2015, 2016, 2017, 2018, 'LT2018',
                        2019, 'LT2019', 'RGBD2019', 'RGBT2019']

    def __init__(self, root_dir, version=2017, anno_type='rect',
                 download=True, return_meta=False, list_file=None):
        super(VOT, self).__init__()
        assert version in self.__valid_versions, 'Unsupport VOT version.'
        assert anno_type in ['default', 'rect'], 'Unknown annotation type.'

        self.root_dir = root_dir
        self.version = version
        self.anno_type = anno_type
        if download:
            self._download(root_dir, version)
        self.return_meta = return_meta

        if list_file is None:
            list_file = os.path.join(root_dir, 'list.txt')
        self._check_integrity(root_dir, version, list_file)

        with open(list_file, 'r') as f:
            self.seq_names = f.read().strip().split('\n')
        self.seq_dirs = [os.path.join(root_dir, s) for s in self.seq_names]
        self.anno_files = [os.path.join(s, 'groundtruth.txt')
                           for s in self.seq_dirs]

    def __getitem__(self, index):
        r"""        
        Args:
            index (integer or string): Index or name of a sequence.
        
        Returns:
            tuple: (img_files, anno) if ``return_meta`` is False, otherwise
                (img_files, anno, meta), where ``img_files`` is a list of
                file names, ``anno`` is a N x 4 (rectangles) or N x 8 (corners) numpy array,
                while ``meta`` is a dict contains meta information about the sequence.
        """
        if isinstance(index, six.string_types):
            if not index in self.seq_names:
                raise Exception('Sequence {} not found.'.format(index))
            index = self.seq_names.index(index)

        img_files = sorted(glob.glob(
            os.path.join(self.seq_dirs[index], 'color', '*.jpg')))
        anno = np.loadtxt(self.anno_files[index], delimiter=',')
        assert len(img_files) == len(anno), (len(img_files), len(anno))
        assert anno.shape[1] in [4, 8]
        if self.anno_type == 'rect' and anno.shape[1] == 8:
            anno = self._corner2rect(anno)

        if self.return_meta:
            meta = self._fetch_meta(
                self.seq_dirs[index], len(img_files))
            return img_files, anno, meta
        else:
            return img_files, anno

    def __len__(self):
        return len(self.seq_names)

    def _download(self, root_dir, version):
        assert version in self.__valid_versions

        if not os.path.isdir(root_dir):
            os.makedirs(root_dir)
        elif os.path.isfile(os.path.join(root_dir, 'list.txt')):
            with open(os.path.join(root_dir, 'list.txt')) as f:
                seq_names = f.read().strip().split('\n')
            if all([os.path.isdir(os.path.join(root_dir, s)) for s in seq_names]):
                print('Files already downloaded.')
                return

        url = 'http://data.votchallenge.net/'
        if version in range(2013, 2015 + 1):
            # main challenge (2013~2015)
            homepage = url + 'vot{}/dataset/'.format(version)
        elif version in range(2015, 2019 + 1):
            # main challenge (2016~2019)
            homepage = url + 'vot{}/main/'.format(version)
        elif version.startswith('LT'):
            # long-term tracking challenge
            year = int(version[2:])
            homepage = url + 'vot{}/longterm/'.format(year)
        elif version.startswith('RGBD'):
            # RGBD tracking challenge
            year = int(version[4:])
            homepage = url + 'vot{}/rgbd/'.format(year)
        elif version.startswith('RGBT'):
            # RGBT tracking challenge
            year = int(version[4:])
            url = url + 'vot{}/rgbtir/'.format(year)
            homepage = url + 'meta/'
        
        # download description file
        bundle_url = homepage + 'description.json'
        bundle_file = os.path.join(root_dir, 'description.json')
        if not os.path.isfile(bundle_file):
            print('Downloading description file...')
            download(bundle_url, bundle_file)

        # read description file
        print('\nParsing description file...')
        with open(bundle_file) as f:
            bundle = json.load(f)

        # md5 generator
        def md5(filename):
            hash_md5 = hashlib.md5()
            with open(filename, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        
        # download all sequences
        seq_names = []
        for seq in bundle['sequences']:
            seq_name = seq['name']
            seq_names.append(seq_name)

            # download channel (color/depth/ir) files
            channels = seq['channels'].keys()
            seq_files = []
            for cn in channels:
                seq_url = seq['channels'][cn]['url']
                if not seq_url.startswith(('http', 'https')):
                    seq_url = url + seq_url[seq_url.find('sequence'):]
                seq_file = os.path.join(
                    root_dir,
                    '{}_{}.zip'.format(seq_name, cn))
                if not os.path.isfile(seq_file) or \
                    md5(seq_file) != seq['channels'][cn]['checksum']:
                    print('\nDownloading %s...' % seq_name)
                    download(seq_url, seq_file)
                seq_files.append(seq_file)

            # download annotations
            anno_url = homepage + '%s.zip' % seq_name
            anno_file = os.path.join(root_dir, seq_name + '_anno.zip')
            if not os.path.isfile(anno_file) or \
                md5(anno_file) != seq['annotations']['checksum']:
                download(anno_url, anno_file)

            # unzip compressed files
            seq_dir = os.path.join(root_dir, seq_name)
            if not os.path.isfile(seq_dir) or len(os.listdir(seq_dir)) < 10:
                print('Extracting %s...' % seq_name)
                os.makedirs(seq_dir)
                for seq_file in seq_files:
                    extract(seq_file, seq_dir)
                extract(anno_file, seq_dir)

        # save list.txt
        list_file = os.path.join(root_dir, 'list.txt')
        with open(list_file, 'w') as f:
            f.write(str.join('\n', seq_names))

        return root_dir

    def _check_integrity(self, root_dir, version, list_file=None):
        assert version in self.__valid_versions
        if list_file is None:
            list_file = os.path.join(root_dir, 'list.txt')

        if os.path.isfile(list_file):
            with open(list_file, 'r') as f:
                seq_names = f.read().strip().split('\n')

            # check each sequence folder
            for seq_name in seq_names:
                seq_dir = os.path.join(root_dir, seq_name)
                if not os.path.isdir(seq_dir):
                    print('Warning: sequence %s not exists.' % seq_name)
        else:
            # dataset not exists
            raise Exception('Dataset not found or corrupted. ' +
                            'You can use download=True to download it.')

    def _corner2rect(self, corners, center=False):
        cx = np.mean(corners[:, 0::2], axis=1)
        cy = np.mean(corners[:, 1::2], axis=1)

        x1 = np.min(corners[:, 0::2], axis=1)
        x2 = np.max(corners[:, 0::2], axis=1)
        y1 = np.min(corners[:, 1::2], axis=1)
        y2 = np.max(corners[:, 1::2], axis=1)

        area1 = np.linalg.norm(corners[:, 0:2] - corners[:, 2:4], axis=1) * \
            np.linalg.norm(corners[:, 2:4] - corners[:, 4:6], axis=1)
        area2 = (x2 - x1) * (y2 - y1)
        scale = np.sqrt(area1 / area2)
        w = scale * (x2 - x1) + 1
        h = scale * (y2 - y1) + 1

        if center:
            return np.array([cx, cy, w, h]).T
        else:
            return np.array([cx - w / 2, cy - h / 2, w, h]).T

    def _fetch_meta(self, seq_dir, frame_num):
        meta = {}

        # attributes
        tag_files = glob.glob(os.path.join(seq_dir, '*.label')) + \
            glob.glob(os.path.join(seq_dir, '*.tag'))
        for f in tag_files:
            tag = os.path.basename(f)
            tag = tag[:tag.rfind('.')]
            meta[tag] = np.loadtxt(f)
        
        # practical
        practical_file = os.path.join(seq_dir, 'practical')
        if os.path.isfile(practical_file + '.value'):
            meta['practical'] = np.loadtxt(practical_file + '.value')
        if os.path.isfile(practical_file + '.txt'):
            meta['practical_txt'] = np.loadtxt(practical_file + '.txt')

        # pad zeros if necessary
        for tag, val in meta.items():
            if len(val) < frame_num:
                meta[tag] = np.pad(
                    val, (0, frame_num - len(val)), 'constant')

        return meta
