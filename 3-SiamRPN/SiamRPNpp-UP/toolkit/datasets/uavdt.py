import os
import json

from tqdm import tqdm
import glob
#from glob import glob
import six
import numpy as np
from .dataset import Dataset
from .video import Video

class UAVDTVideo(Video):
    """
    Args:
        name: video name
        root: dataset root
        video_dir: video directory
        init_rect: init rectangle
        img_names: image names
        gt_rect: groundtruth rectangle
        attr: attribute of video
    """
    def __init__(self, name, root, video_dir, init_rect, img_names,
            gt_rect, attr, load_img=False):
        super(UAVDTVideo, self).__init__(name, root, video_dir,
                init_rect, img_names, gt_rect, attr, load_img)

class UAVDTDataset(Dataset):
    """
    Args:
        name: dataset name, should be 'UAV123', 'UAV20L'
        dataset_root: dataset root
        load_img: wether to load all imgs
    """
    def __init__(self, name, dataset_root, load_img=False):
        super(UAVDTDataset, self).__init__(name, dataset_root)
        # with open(os.path.join(dataset_root, name+'.json'), 'r') as f:
        #     meta_data = json.load(f)
        #self.root_dir = dataset_root

        self._check_integrity(dataset_root)

        self.anno_files = sorted(glob.glob(
            os.path.join(dataset_root, '*/*_gt.txt')))
        self.seq_dirs = [os.path.dirname(f) for f in self.anno_files]
        self.seq_names = [os.path.basename(d) for d in self.seq_dirs]
        
        meta_data=dict()

        for  i in range(len(self.seq_names)):
            img_names = sorted(glob.glob(
            os.path.join(self.seq_dirs[i], '*.jpg')))
            img_names=[x.split('/UAVDT/')[-1] for x in img_names]
            gt_rect = np.loadtxt(self.anno_files[i], delimiter=',')
            data=dict()
            data['video_dir']=self.seq_names[i]
            data['init_rect']=gt_rect[0]
            data['img_names']=img_names
            data['gt_rect']=gt_rect
            data['attr']=0
            meta_data[self.seq_names[i]]=data

        # load videos
        pbar = tqdm(meta_data.keys(), desc='loading '+name, ncols=100)
        self.videos = {}
        for  video in pbar:
           #pbar.set_postfix_str(video)
            self.videos[video] = UAVDTVideo(video,
                                          dataset_root,
                                          meta_data[video]['video_dir'],
                                          meta_data[video]['init_rect'],
                                          meta_data[video]['img_names'],
                                          meta_data[video]['gt_rect'],
                                          meta_data[video]['attr'])
        # set attr  这哪是不考虑属性
        # attr = []
        # for x in self.videos.values():
        #     attr += x.attr
        # attr = set(attr)
        # self.attr = {}
        # self.attr['ALL'] = list(self.videos.keys())
        # for x in attr:
        #     self.attr[x] = []
        # for k, v in self.videos.items():
        #     for attr_ in v.attr:
        #         self.attr[attr_].append(k)

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

