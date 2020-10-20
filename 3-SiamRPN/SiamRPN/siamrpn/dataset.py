import torch
import cv2
import os
import numpy as np
import pickle
import lmdb
import hashlib
import glob
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

from torch.utils.data.dataset import Dataset
from .generate_anchors import generate_anchors
from .config import config
from .utils import box_transform, compute_iou, add_box_img, crop_and_pad

from IPython import embed


class ImagnetVIDDataset(Dataset):
    def __init__(self, db, video_names, data_dir, z_transforms, x_transforms, training=True):
        self.video_names = video_names
        self.data_dir = data_dir
        self.z_transforms = z_transforms
        self.x_transforms = x_transforms
        meta_data_path = os.path.join(data_dir, 'meta_data.pkl')
        self.meta_data = pickle.load(open(meta_data_path, 'rb'))
        self.meta_data = {x[0]: x[1] for x in self.meta_data} # 181402（meta_data的len） --> 178513(视频文件夹的数量)？？
        # filter traj len less than 2
        for key in self.meta_data.keys(): #key是视频文件名字
            trajs = self.meta_data[key]
            for trkid in list(trajs.keys()):
                if len(trajs[trkid]) < 2:
                    del trajs[trkid]

        self.txn = db.begin(write=False)
        self.num = len(self.video_names) if config.pairs_per_video_per_epoch is None or not training \
            else config.pairs_per_video_per_epoch * len(self.video_names)#训练模式下数据量为什么翻倍
       
        # data augmentation  
        self.max_stretch = config.scale_resize       #0.15
        self.max_translate = config.max_translate    #12
        self.random_crop_size = config.instance_size #271
        self.center_crop_size = config.exemplar_size #127

        self.training = training  #True

        valid_scope = 2 * config.valid_scope + 1#config.valid_scope=9
        self.anchors = generate_anchors(config.total_stride, config.anchor_base_size, config.anchor_scales,
                                        config.anchor_ratios,
                                        valid_scope)

    def imread(self, path):
        key = hashlib.md5(path.encode()).digest()
        img_buffer = self.txn.get(key)
        img_buffer = np.frombuffer(img_buffer, np.uint8)
        img = cv2.imdecode(img_buffer, cv2.IMREAD_COLOR)
        return img

    def _sample_weights(self, center, low_idx, high_idx, s_type='uniform'):
        weights = list(range(low_idx, high_idx))
        weights.remove(center)
        weights = np.array(weights)
        if s_type == 'linear':
            weights = abs(weights - center)
        elif s_type == 'sqrt':
            weights = np.sqrt(abs(weights - center))
        elif s_type == 'uniform':
            weights = np.ones_like(weights)
        return weights / sum(weights)

    def RandomStretch(self, sample, gt_w, gt_h):
        scale_h = 1.0 + np.random.uniform(-self.max_stretch, self.max_stretch)
        scale_w = 1.0 + np.random.uniform(-self.max_stretch, self.max_stretch)
        h, w = sample.shape[:2]
        shape = int(w * scale_w), int(h * scale_h)
        scale_w = int(w * scale_w) / w
        scale_h = int(h * scale_h) / h
        gt_w = gt_w * scale_w
        gt_h = gt_h * scale_h
        return cv2.resize(sample, shape, cv2.INTER_LINEAR), gt_w, gt_h

    def compute_target(self, anchors, box):
        regression_target = box_transform(anchors, box)#box=[gt_cx,gt_cy,gt_w,gt_h]，regression—target 回归值offset
                            
        iou = compute_iou(anchors, box).flatten()#1805个iou
        # print(np.max(iou))
        pos_index = np.where(iou > config.pos_threshold)[0] #返回大于0.6的index，作为正样本索引
        neg_index = np.where(iou < config.neg_threshold)[0] #返回小于0.3的index，作为负样本索引
        label = np.ones_like(iou) * -1 #大于0.6等于1； 小于0.3等于0； 介于0.3和0.6之间的数等于-1
        label[pos_index] = 1
        label[neg_index] = 0
        return regression_target, label

    def __getitem__(self, idx):

        all_idx = np.arange(self.num)#参数连续的数字
        np.random.shuffle(all_idx)#随机打乱
        all_idx = np.insert(all_idx, 0, idx, 0)#数组，需要插入的位置，需要插入的数值，指定插入的维度 插入为何？？
        for idx in all_idx:
            idx = idx % len(self.video_names)#179587
            video = self.video_names[idx]
            trajs = self.meta_data[video] #178513 meta_data和 video_names数量对不上
            # sample one trajs
            if len(trajs.keys()) == 0:
                continue

            trkid = np.random.choice(list(trajs.keys()))#随机选择一个
            traj = trajs[trkid]
            assert len(traj) > 1, "video_name: {}".format(video)
            # sample exemplar
            exemplar_idx = np.random.choice(list(range(len(traj))))
            # exemplar_name = os.path.join(self.data_dir, video, traj[exemplar_idx] + ".{:02d}.x*.jpg".format(trkid))
            
            str=os.path.join(self.data_dir, video)

            if 'ILSVRC2015' in video: #对于ILSVRC——VID数据的读取
                exemplar_name = glob.glob(os.path.join(self.data_dir, video, traj[exemplar_idx] + ".{:02d}.x*.jpg".format(trkid)))[0]
            else:#对于ytb数据的读取
                if os.path.exists(str):
                    exemplar_name = glob.glob(os.path.join(self.data_dir, video, traj[exemplar_idx] + ".{}.x*.jpg".format(trkid)))[0]
                else:
                    print(video)
            exemplar_gt_w, exemplar_gt_h, exemplar_w_image, exemplar_h_image = \
                float(exemplar_name.split('_')[-4]), float(exemplar_name.split('_')[-3]), \
                float(exemplar_name.split('_')[-2]), float(exemplar_name.split('_')[-1][:-4])

            exemplar_ratio = min(exemplar_gt_w / exemplar_gt_h, exemplar_gt_h / exemplar_gt_w)

            exemplar_scale = exemplar_gt_w * exemplar_gt_h / (exemplar_w_image * exemplar_h_image)
            #尺度exemplar_scale在0.001和0.7之间；纵横比exemplar_ratio0.1和10之间
            if not config.scale_range[0] <= exemplar_scale < config.scale_range[1]:
                continue
            if not config.ratio_range[0] <= exemplar_ratio < config.ratio_range[1]:
                continue

            exemplar_img = self.imread(exemplar_name)
            # exemplar_img = cv2.cvtColor(exemplar_img, cv2.COLOR_BGR2RGB)
            # sample instance
            if 'ILSVRC2015' in exemplar_name:
                frame_range = config.frame_range_vid #config.frame_range_vid=100
            else:
                frame_range = config.frame_range_ytb #config.frame_range_ytb=1 ？？？ 为何ytb的范围小
            low_idx = max(0, exemplar_idx - frame_range)
            up_idx = min(len(traj), exemplar_idx + frame_range + 1)

            # create sample weight, if the sample are far away from center
            # the probability being choosen are high
            weights = self._sample_weights(exemplar_idx, low_idx, up_idx, config.sample_type)#config.sample_type=‘uniform’所以样本权重是一样的
            instance = np.random.choice(traj[low_idx:exemplar_idx] + traj[exemplar_idx + 1:up_idx], p=weights)
            #随机选择一个实例序列
            if 'ILSVRC2015' in video:
                instance_name = glob.glob(os.path.join(self.data_dir, video, instance + ".{:02d}.x*.jpg".format(trkid)))[0]
            else:
                instance_name = glob.glob(os.path.join(self.data_dir, video, instance + ".{}.x*.jpg".format(trkid)))[0]

            instance_gt_w, instance_gt_h, instance_w_image, instance_h_image = \
                float(instance_name.split('_')[-4]), float(instance_name.split('_')[-3]), \
                float(instance_name.split('_')[-2]), float(instance_name.split('_')[-1][:-4])
            instance_ratio = min(instance_gt_w / instance_gt_h, instance_gt_h / instance_gt_w)
            instance_scale = instance_gt_w * instance_gt_h / (instance_w_image * instance_h_image)
            if not config.scale_range[0] <= instance_scale < config.scale_range[1]:
                continue
            if not config.ratio_range[0] <= instance_ratio < config.ratio_range[1]:
                continue

            instance_img = self.imread(instance_name)
            # instance_img = cv2.cvtColor(instance_img, cv2.COLOR_BGR2RGB)

            if np.random.rand(1) < config.gray_ratio: #转换成灰度，再转换成彩色？？
                exemplar_img = cv2.cvtColor(exemplar_img, cv2.COLOR_RGB2GRAY) 
                exemplar_img = cv2.cvtColor(exemplar_img, cv2.COLOR_GRAY2RGB)
                instance_img = cv2.cvtColor(instance_img, cv2.COLOR_RGB2GRAY)
                instance_img = cv2.cvtColor(instance_img, cv2.COLOR_GRAY2RGB)
            if config.exem_stretch:#False 这里并没有进行伸展操作 
                exemplar_img, exemplar_gt_w, exemplar_gt_h = self.RandomStretch(exemplar_img, exemplar_gt_w,
                                                                                exemplar_gt_h)
            exemplar_img, _ = crop_and_pad(exemplar_img, (exemplar_img.shape[1] - 1) / 2,
                                           (exemplar_img.shape[0] - 1) / 2, self.center_crop_size,
                                           self.center_crop_size)

            # exemplar_img_np = exemplar_img.copy()

            instance_img, gt_w, gt_h = self.RandomStretch(instance_img, instance_gt_w, instance_gt_h)
            im_h, im_w, _ = instance_img.shape
            cy_o = (im_h - 1) / 2 
            cx_o = (im_w - 1) / 2
            cy = cy_o + np.random.randint(- self.max_translate, self.max_translate + 1)
            cx = cx_o + np.random.randint(- self.max_translate, self.max_translate + 1)
            gt_cx = cx_o - cx #？
            gt_cy = cy_o - cy #？

            instance_img_1, scale = crop_and_pad(instance_img, cx, cy, self.random_crop_size, self.random_crop_size)
            exemplar_img = self.z_transforms(exemplar_img)

            instance_img_1 = self.x_transforms(instance_img_1)
            #regression_target是交并比，conf_target=
            regression_target, conf_target = self.compute_target(self.anchors,np.array(list(map(round, [gt_cx, gt_cy, gt_w, gt_h]))))
            
            return exemplar_img, instance_img_1, regression_target, conf_target.astype(np.int64)
            #


    def draw_img(self, img, boxes, name='1.jpg', color=(0, 255, 0)):
        # boxes (x,y,w,h)
        img = img.copy()
        img_ctx = (img.shape[1] - 1) / 2
        img_cty = (img.shape[0] - 1) / 2
        for box in boxes:
            point_1 = img_ctx - box[2] / 2 + box[0], img_cty - box[3] / 2 + box[1]
            point_2 = img_ctx + box[2] / 2 + box[0], img_cty + box[3] / 2 + box[1]
            img = cv2.rectangle(img, (int(point_1[0]), int(point_1[1])), (int(point_2[0]), int(point_2[1])),
                                color, 2)
        cv2.imwrite(name, img)

    def __len__(self):
        return self.num
