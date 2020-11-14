import os
from collections import defaultdict
from glob import glob

import numpy as np
from PIL import Image


class DAVIS(object):
    SUBSET_OPTIONS = ['train', 'val', 'test-dev', 'test-challenge']
    TASKS = ['semi-supervised', 'unsupervised']
    DATASET_WEB = 'https://davischallenge.org/davis2017/code.html'
    VOID_LABEL = 255

    def __init__(self,
                 root,
                 task='unsupervised',
                 subset='val',
                 sequences='all',
                 resolution='480p',
                 codalab=False,
                 version='2017'):
        """
        Class to read the DAVIS dataset
        :param root: Path to the DAVIS folder that contains JPEGImages, Annotations, etc. folders.
        :param task: Task to load the annotations, choose between semi-supervised or unsupervised.
        :param subset: Set to load the annotations
        :param sequences: Sequences to consider, 'all' to use all the sequences in a set.
        :param resolution: Specify the resolution to use the dataset, choose between '480' and 'Full-Resolution'
        """
        self.version = version
        print('-' * 10)
        print(version)
        print('-' * 10)

        if subset not in self.SUBSET_OPTIONS:
            raise ValueError(
                'Subset should be in {}'.format('self.SUBSET_OPTIONS'))
        if task not in self.TASKS:
            raise ValueError(
                'The only tasks that are supported are {}'.format('self.TASKS'))

        self.task = task
        self.subset = subset
        self.root = root
        self.img_path = os.path.join(self.root, 'JPEGImages', resolution)
        annotations_folder = 'Annotations' if task == 'semi-supervised' else 'Annotations_unsupervised'
        if version == '2017':
            self.mask_path = os.path.join(self.root, annotations_folder,
                                          resolution)
        else:
            self.mask_path = os.path.join(self.root, annotations_folder,
                                          resolution + '_2016')

        year = '2019' if task == 'unsupervised' and (
            subset == 'test-dev' or subset == 'test-challenge') else '2017'

        self.imagesets_path = os.path.join(self.root, 'ImageSets', year)

        self._check_directories()

        if sequences == 'all':
            txt_path = os.path.join(self.imagesets_path,
                                    '{}.txt'.format(self.subset))
            txt_path = txt_path.replace('2017', version)
            #print(txt_path, version)
            with open(txt_path, 'r') as f:
                tmp = f.readlines()
            sequences_names = [x.strip() for x in tmp]
        else:
            sequences_names = sequences if isinstance(sequences,
                                                      list) else [sequences]
        self.sequences = defaultdict(dict)

        for seq in sequences_names:
            images = np.sort(glob(os.path.join(self.img_path, seq,
                                               '*.jpg'))).tolist()
            if len(images) == 0 and not codalab:
                raise FileNotFoundError(
                    'Images for sequence {} not found.'.format(seq))
            self.sequences[seq]['images'] = images
            masks = np.sort(glob(os.path.join(self.mask_path, seq,
                                              '*.png'))).tolist()
            masks.extend([-1] * (len(images) - len(masks)))
            self.sequences[seq]['masks'] = masks

    def _check_directories(self):
        if not os.path.exists(self.root):
            raise FileNotFoundError(
                'DAVIS not found in the specified directory, download it from {}'
                .format('self.DATASET_WEB'))
        if not os.path.exists(
                os.path.join(self.imagesets_path, '{}.txt'.format(
                    self.subset))):
            pass
            #raise FileNotFoundError('Subset sequences list for {} not found, download the missing subset '.format('self.subset')
            #                        'for the {} task from {self.DATASET_WEB}'.format(''))
        if self.subset in ['train', 'val'
                           ] and not os.path.exists(self.mask_path):
            raise FileNotFoundError(
                'Annotations folder for the {} task not found, download it from {}'
                .format(self.task, self.DATASET_WEB))

    def get_frames(self, sequence):
        for img, msk in zip(self.sequences[sequence]['images'],
                            self.sequences[sequence]['masks']):
            image = np.array(Image.open(img))
            mask = None if msk is None else np.array(Image.open(msk))
            yield image, mask

    def _get_all_elements(self, sequence, obj_type):

        #print(self.sequences[sequence][obj_type])
        obj = np.array(Image.open(self.sequences[sequence][obj_type][0]))
        #print(obj, obj.shape)
        flag = 0
        if obj.shape[-1] == 3:
            flag = 1
            obj = obj[:, :, 0]

        all_objs = np.zeros(
            (len(self.sequences[sequence][obj_type]), *obj.shape))
        obj_id = []
        for i, obj in enumerate(self.sequences[sequence][obj_type]):
            if flag == 0:
                all_objs[i, ...] = np.array(Image.open(obj))
            else:
                all_objs[i, ...] = np.array(Image.open(obj))[:, :, 0]
            obj_id.append(''.join(obj.split('/')[-1].split('.')[:-1]))
        return all_objs, obj_id

    def get_all_images(self, sequence):
        return self._get_all_elements(sequence, 'images')

    def get_all_masks(self, sequence, separate_objects_masks=False):
        masks, masks_id = self._get_all_elements(sequence, 'masks')
        masks_void = np.zeros_like(masks)

        # Separate void and object masks
        for i in range(masks.shape[0]):
            masks_void[i, ...] = masks[i, ...] == 255
            masks[i, masks[i, ...] == 255] = 0

        if separate_objects_masks:
            num_objects = int(np.max(masks[0, ...]))
            tmp = np.ones((num_objects, *masks.shape))
            tmp = tmp * np.arange(1, num_objects + 1)[:, None, None, None]
            masks = (tmp == masks[None, ...])
            masks = masks > 0
        return masks, masks_void, masks_id

    def get_sequences(self):
        for seq in self.sequences:
            yield seq


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    only_first_frame = True
    subsets = ['train', 'val']

    for s in subsets:
        dataset = DAVIS(
            root='/home/csergi/scratch2/Databases/DAVIS2017_private', subset=s)
        for seq in dataset.get_sequences():
            g = dataset.get_frames(seq)
            img, mask = next(g)
            plt.subplot(2, 1, 1)
            plt.title(seq)
            plt.imshow(img)
            plt.subplot(2, 1, 2)
            plt.imshow(mask)
            plt.show(block=True)
