
import json
import os

from tqdm import tqdm

from .dataset import Dataset
from .video import Video
import glob 

import numpy as np 

class GOT10kVideo(Video):
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
        super(GOT10kVideo, self).__init__(name, root, video_dir,
                init_rect, img_names, gt_rect, attr, load_img)

    # def load_tracker(self, path, tracker_names=None):
    #     """
    #     Args:
    #         path(str): path to result
    #         tracker_name(list): name of tracker
    #     """
    #     if not tracker_names:
    #         tracker_names = [x.split('/')[-1] for x in glob(path)
    #                 if os.path.isdir(x)]
    #     if isinstance(tracker_names, str):
    #         tracker_names = [tracker_names]
    #     # self.pred_trajs = {}
    #     for name in tracker_names:
    #         traj_file = os.path.join(path, name, self.name+'.txt')
    #         if os.path.exists(traj_file):
    #             with open(traj_file, 'r') as f :
    #                 self.pred_trajs[name] = [list(map(float, x.strip().split(',')))
    #                         for x in f.readlines()]
    #             if len(self.pred_trajs[name]) != len(self.gt_traj):
    #                 print(name, len(self.pred_trajs[name]), len(self.gt_traj), self.name)
    #         else:

    #     self.tracker_names = list(self.pred_trajs.keys())

class GOT10kDataset(Dataset):
    """
    Args:
        name:  dataset name, should be "NFS30" or "NFS240"
        dataset_root, dataset root dir
    """
    def __init__(self, name, dataset_root, load_img=False):
        super(GOT10kDataset, self).__init__(name, dataset_root)
        # with open(os.path.join(dataset_root, name+'.json'), 'r') as f:
        #     meta_data = json.load(f)
        
        #self._check_integrity(dataset_root)

        self.anno_files = sorted(glob.glob(
            os.path.join(dataset_root, 'val/*/*.txt')))
        self.seq_dirs = [os.path.dirname(f) for f in self.anno_files]
        self.seq_names = [os.path.basename(d) for d in self.seq_dirs]
        
        meta_data=dict()

        for  i in range(len(self.seq_names)):
            img_names = sorted(glob.glob(
            os.path.join(self.seq_dirs[i], '*.jpg')))
            img_names=[x.split('/GOT-10k/')[-1] for x in img_names]
            gt_rect = np.loadtxt(self.anno_files[i], delimiter=',')
            data=dict()
            data['video_dir']=self.seq_names[i]
            data['init_rect']=gt_rect[0]
            data['img_names']=img_names
            data['gt_rect']=gt_rect
            # data['attr']=0
            meta_data[self.seq_names[i]]=data


        # load videos
        pbar = tqdm(meta_data.keys(), desc='loading '+name, ncols=100)
        self.videos = {}
        for video in pbar:
            pbar.set_postfix_str(video)
            self.videos[video] = GOT10kVideo(video,
                                          dataset_root,
                                          meta_data[video]['video_dir'],
                                          meta_data[video]['init_rect'],
                                          meta_data[video]['img_names'],
                                          meta_data[video]['gt_rect'],
                                          None)
        self.attr = {}
        self.attr['ALL'] = list(self.videos.keys())
