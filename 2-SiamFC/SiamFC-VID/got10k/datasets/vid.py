from __future__ import absolute_import, print_function

import os
import glob
import six
import numpy as np
import xml.etree.ElementTree as ET
import json
from collections import OrderedDict


class ImageNetVID(object):
    r"""`ImageNet Video Image Detection (VID) <https://image-net.org/challenges/LSVRC/2015/#vid>`_ Dataset.

    Publication:
        ``ImageNet Large Scale Visual Recognition Challenge``, O. Russakovsky,
            J. deng, H. Su, etc. IJCV, 2015.
    
    Args:
        root_dir (string): Root directory of dataset where ``Data``, and
            ``Annotation`` folders exist.
        subset (string, optional): Specify ``train``, ``val`` or (``train``, ``val``)
            subset(s) of ImageNet-VID. Default is a tuple (``train``, ``val``).
        cache_dir (string, optional): Directory for caching the paths and annotations
            for speeding up loading. Default is ``cache/imagenet_vid``.
    """
    def __init__(self, root_dir, subset=('train', 'val'),
                 cache_dir='cache/imagenet_vid'):
        self.root_dir = root_dir
        self.cache_dir = cache_dir
        if isinstance(subset, str):
            assert subset in ['train', 'val']
            self.subset = [subset]
        elif isinstance(subset, (list, tuple)):
            assert all([s in ['train', 'val'] for s in subset])
            self.subset = subset
        else:
            raise Exception('Unknown subset')
        
        # cache filenames and annotations to speed up training
        self.seq_dict = self._cache_meta()
        self.seq_names = [n for n in self.seq_dict]

    def __getitem__(self, index):
        r"""        
        Args:
            index (integer or string): Index or name of a sequence.
        
        Returns:
            tuple: (img_files, anno), where ``img_files`` is a list of
                file names and ``anno`` is a N x 4 (rectangles) numpy array.
        """
        if isinstance(index, six.string_types):
            seq_name = index
        else:
            seq_name = self.seq_names[index]

        seq_dir, frames, anno_file = self.seq_dict[seq_name]
        img_files = [os.path.join(
            seq_dir, '%06d.JPEG' % f) for f in frames]
        anno = np.loadtxt(anno_file, delimiter=',')

        return img_files, anno

    def __len__(self):
        return len(self.seq_dict)

    def _cache_meta(self):
        cache_file = os.path.join(self.cache_dir, 'seq_dict.json')
        if os.path.isfile(cache_file):
            print('Dataset already cached.')
            with open(cache_file) as f:
                seq_dict = json.load(f, object_pairs_hook=OrderedDict)
            return seq_dict
        
        # image and annotation paths
        print('Gather sequence paths...')
        seq_dirs = []
        anno_dirs = []
        if 'train' in self.subset:
            seq_dirs_ = sorted(glob.glob(os.path.join(
                self.root_dir, 'Data/VID/train/ILSVRC*/ILSVRC*')))
            anno_dirs_ = [os.path.join(
                self.root_dir, 'Annotations/VID/train',
                *s.split('/')[-2:]) for s in seq_dirs_]
            seq_dirs += seq_dirs_
            anno_dirs += anno_dirs_
        if 'val' in self.subset:
            seq_dirs_ = sorted(glob.glob(os.path.join(
                self.root_dir, 'Data/VID/val/ILSVRC2015_val_*')))
            anno_dirs_ = [os.path.join(
                self.root_dir, 'Annotations/VID/val',
                s.split('/')[-1]) for s in seq_dirs_]
            seq_dirs += seq_dirs_
            anno_dirs += anno_dirs_
        seq_names = [os.path.basename(s) for s in seq_dirs]

        # cache paths and annotations
        print('Caching annotations to %s, ' % self.cache_dir + \
            'it may take a few minutes...')
        seq_dict = OrderedDict()
        cache_anno_dir = os.path.join(self.cache_dir, 'anno')
        if not os.path.isdir(cache_anno_dir):
            os.makedirs(cache_anno_dir)

        for s, seq_name in enumerate(seq_names):
            if s % 100 == 0 or s == len(seq_names) - 1:
                print('--Caching sequence %d/%d: %s' % \
                    (s + 1, len(seq_names), seq_name))
            anno_files = sorted(glob.glob(os.path.join(
                anno_dirs[s], '*.xml')))
            objects = [ET.ElementTree(file=f).findall('object')
                       for f in anno_files]
            
            # find all track ids
            track_ids, counts = np.unique([
                obj.find('trackid').text for group in objects
                for obj in group], return_counts=True)
            
            # fetch paths and annotations for each track id
            for t, track_id in enumerate(track_ids):
                if counts[t] < 2:
                    continue
                frames = []
                anno = []
                for f, group in enumerate(objects):
                    for obj in group:
                        if not obj.find('trackid').text == track_id:
                            continue
                        frames.append(f)
                        anno.append([
                            int(obj.find('bndbox/xmin').text),
                            int(obj.find('bndbox/ymin').text),
                            int(obj.find('bndbox/xmax').text),
                            int(obj.find('bndbox/ymax').text)])
                anno = np.array(anno, dtype=int)
                anno[:, 2:] -= anno[:, :2] - 1

                # store annotations
                key = '%s.%d' % (seq_name, int(track_id))
                cache_anno_file = os.path.join(cache_anno_dir, key + '.txt')
                np.savetxt(cache_anno_file, anno, fmt='%d', delimiter=',')

                # store paths
                seq_dict.update([(key, [
                    seq_dirs[s], frames, cache_anno_file])])
        
        # store seq_dict
        with open(cache_file, 'w') as f:
            json.dump(seq_dict, f)

        return seq_dict
