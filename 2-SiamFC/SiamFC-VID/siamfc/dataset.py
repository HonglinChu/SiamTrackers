import torch
import cv2
import os
import numpy as np
import pickle
import lmdb
import hashlib
from torch.utils.data.dataset import Dataset

from .config import config

class ImagnetVIDDataset(Dataset):
    def __init__(self, db, video_names, data_dir, z_transforms, x_transforms, training=True):
    #def __init__(self, video_names, data_dir, z_transforms, x_transforms, training=True):
        self.video_names = video_names  #所有视频的名字
        self.data_dir = data_dir        #视频所在的文件夹路径
        self.z_transforms = z_transforms 
        self.x_transforms = x_transforms
        meta_data_path = os.path.join(data_dir, 'meta_data.pkl')
        self.meta_data = pickle.load(open(meta_data_path, 'rb')) # 包括每一个视频名字和对应的所有图片号
        self.meta_data = {x[0]:x[1] for x in self.meta_data}     # x[0]:x[1] --> {'name1':seq1,'name2':seq2 }
        # filter traj len less than 2  一个视频文件夹里面可能包含多个视频序列如 ILSVRC2015_train_00133002视频文件夹(每一张图片可能有多个目标)，在这里删除序列小于2得 
        for key in self.meta_data.keys(): #  .keys() 以列表返回一个dict_keys对象，包含所有的键 ，如['name1', 'name2']
            trajs = self.meta_data[key]   #所有的图像序列
            for trkid in list(trajs.keys()):
                if len(trajs[trkid]) < 2:
                    del trajs[trkid]

        self.txn = db.begin(write=False) #禁止写操作
       
        self.num = len(self.video_names) if config.num_per_epoch is None or not training\
                else config.num_per_epoch

    def imread(self, path):
        #lmdb 读取高效
        key = hashlib.md5(path.encode()).digest()#将路径转换成对应的键值
        img_buffer = self.txn.get(key)#通过键值查询对应的图像  
        img_buffer = np.frombuffer(img_buffer, np.uint8) #通过这个二进编码获得图像一维的ndarray数组信息
        img = cv2.imdecode(img_buffer, cv2.IMREAD_COLOR) ##通过数组进行图像获取图像

        #cv2.imread读取速度非常慢
        #img=cv2.imread(path, cv2.IMREAD_COLOR)
        return img
    #创建样本权重， 函数名前面下杠 '_' 代表私有函数； 假设center（exemplar_id）=200,在没有超出序列总长度的情况下则 low-idx=100，high_idx=300 
    def _sample_weights(self, center, low_idx, high_idx, s_type='uniform'): 
        weights = list(range(low_idx, high_idx))# 100-300之间的序列
        weights.remove(center)  #  移除 exemplar_id，即 center_id
        weights = np.array(weights) # 转换成numpy数组
        if s_type == 'linear':
            weights = abs(weights - center)
        elif s_type == 'sqrt':
            weights = np.sqrt(abs(weights - center))
        elif s_type == 'uniform':
            weights = np.ones_like(weights)
        return weights / sum(weights)

    def __getitem__(self, idx):

        idx = idx % len(self.video_names)
        video = self.video_names[idx]  # 视频序列
        trajs = self.meta_data[video]  # 可能包含了多个视频序列 meta_data中每一个视频可能包含很多个视频序列（多个目标，每个目标对应一个视频序列）
        # sample one trajs     关于np.random.choice的使用 https://blog.csdn.net/ImwaterP/article/details/96282230
        trkid = np.random.choice(list(trajs.keys())) #从多个目标序列中选择一个序列id
        traj = trajs[trkid]#选择id为trkid的序列

        assert len(traj) > 1, "video_name: {}".format(video) #判断如果traj序列长度大于1,

        # sample exemplar  
        exemplar_idx = np.random.choice(list(range(len(traj)))) #从视频序列中选择一个图片id，作为exemplar_idx
        exemplar_name= os.path.join(self.data_dir, video, traj[exemplar_idx]+".{:02d}.x.jpg".format(trkid))#获得对应的图片路径名
        exemplar_img = self.imread(exemplar_name)
        exemplar_img = cv2.cvtColor(exemplar_img, cv2.COLOR_BGR2RGB) # RGB色彩颜色转换
        # sample instance
        low_idx = max(0, exemplar_idx - config.frame_range)
        up_idx = min(len(traj), exemplar_idx + config.frame_range)   # 0<= low_id <  exemplar_idx < up_id<=exemplar_idx+100

        # create sample weight, if the sample are far away from center
        # the probability being choosen are high，在这里config.sample_type='uniform'代表每个样本的权重是一样的
        weights = self._sample_weights(exemplar_idx, low_idx, up_idx, config.sample_type) 
        instance = np.random.choice(traj[low_idx:exemplar_idx] + traj[exemplar_idx+1:up_idx], p=weights)#p用来规定选取列表中每个元素的概率，默认概率是相同的
        instance_name = os.path.join(self.data_dir, video, instance+".{:02d}.x.jpg".format(trkid))
        instance_img = self.imread(instance_name)
        instance_img = cv2.cvtColor(instance_img, cv2.COLOR_BGR2RGB)
        if np.random.rand(1) < config.gray_ratio: #  np.random.rand(1)  返回一个0-1均匀分布的随机数字。
            exemplar_img = cv2.cvtColor(exemplar_img, cv2.COLOR_RGB2GRAY)
            exemplar_img = cv2.cvtColor(exemplar_img, cv2.COLOR_GRAY2RGB)#？？
            instance_img = cv2.cvtColor(instance_img, cv2.COLOR_RGB2GRAY)
            instance_img = cv2.cvtColor(instance_img, cv2.COLOR_GRAY2RGB)#？？
        exemplar_img = self.z_transforms(exemplar_img) #抠图 127x127
        instance_img = self.x_transforms(instance_img) #抠图 （255-8x2）x（255-8x2）
        return exemplar_img, instance_img

    def __len__(self):
        return self.num
