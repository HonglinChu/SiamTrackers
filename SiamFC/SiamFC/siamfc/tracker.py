from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import cv2
import sys
import os
from collections import namedtuple
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from got10k.trackers import Tracker   # GOT-10k

from .utils import init_weights,crop_and_resize,read_image,show_image,load_pretrain
from .backbones import AlexNetV0, AlexNetV1,ResNet22,ResNeXt22,ResNet22W
from .heads import SiamFC
from .losses import BalancedLoss
from .datasets import Pair
from .transforms import SiamFCTransforms
from .network import SiamFCNet 
from .config import config
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler  #使用 DistributedSampler 对数据集进行划分,保证每一个batch的数据被分摊到每一个进程上,每个进程读取不同数据
from torch.nn.parallel import DistributedDataParallel
from .utils import get_logger 

__all__ = ['SiamFCTracker']

class SiamFCTracker(Tracker):  #定义一个追踪器

    def __init__(self,model_path=None,cfg=None,group=2):
        super(SiamFCTracker, self).__init__(model_path, True)
        
        self.cfg=config  
        
        # 在这里更新参数,主要是超参数寻优的时候用到
        if cfg:
            config.update(cfg) 
        # setup model
        #(1)分组卷积的时候不能加载预训练模型

        if group==2:
            self.net = SiamFCNet(backbone=AlexNetV1(),  head=SiamFC(self.cfg.out_scale))
        # (2)没有使用分组卷积,可以加载预训练模型alexnet
        else:
            self.net = SiamFCNet(backbone=AlexNetV0(),  head=SiamFC(self.cfg.out_scale))

        init_weights(self.net) #对网络权重进行初始化
        
        # load checkpoint if provided  ,group的时候分组卷积,不能加载alexnet 
        if model_path is not None:       #加载训练好的网络模型
            if 'alexnet' in model_path and group==1:  # group=2 时候不能加载alexnet
                self.net = load_pretrain(self.net, model_path)  # load pretrain
            elif 'siamfc' in model_path: #加载checkpoint
                self.net.load_state_dict(torch.load(
                model_path, map_location=lambda storage, loc: storage))

        #self.net = self.net.to(self.device) #将模型加载到GPU上
        self.net=self.net.cuda()
        
        # setup criterion
        self.criterion = BalancedLoss()

        # setup optimizer
        self.optimizer = optim.SGD(
            self.net.parameters(),
            lr=config.initial_lr,
            weight_decay=config.weight_decay,
            momentum=config.momentum)         
        
        # setup lr scheduler  #动态调整学习率,这个是怎么计算的？？
        gamma = np.power(config.ultimate_lr / config.initial_lr, 1.0 / config.epoch_num)

        #gamma=0.87
        self.lr_scheduler = ExponentialLR(self.optimizer, gamma)
  
    #namedtuple比tuple更强大，与list不同的是,你不能改变tuple中元素的数值 
    #Namedtuple比普通tuple具有更好的可读性，可以使代码更易于维护
    #为了构造一个namedtuple需要两个参数，分别是tuple的名字和其中域的名字
     
    '''禁止计算局部梯度
    方法1 使用装饰器 @torch.no_gard()修饰的函数，在调用时不允许计算梯度
    方法2 # 将不用计算梯度的变量放在 with torch.no_grad()里
    '''
    @torch.no_grad()#
    def init(self, img, box):
        # set to evaluation mode
        self.net.eval() 

        # convert box to 0-indexed and center based [y, x, h, w]
        box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32)
        self.center, self.target_sz = box[:2], box[2:] #最原始的图片大小 从groundtruth读取

        # create hanning window  response_up=16 ；  response_sz=17 ； self.upscale_sz=272
        self.upscale_sz = config.response_up * config.response_sz  
        self.hann_window = np.outer(  # np.outer 如果a，b是高维数组，函数会自动将其flatten成1维 ，用来求外积
            np.hanning(self.upscale_sz),
            np.hanning(self.upscale_sz))
        self.hann_window /= self.hann_window.sum()  #？？？

        # search scale factors
        self.scale_factors = config.scale_step ** np.linspace( #linspace 在start和stop之间返回均匀间隔的数据
            -(config.scale_num // 2), #//py3中双斜杠代表向下取整
            config.scale_num // 2, config.scale_num)
        
        # exemplar and search sizes  config.context=1/2  
        context = config.context * np.sum(self.target_sz)    # 引入margin：2P=(长+宽）× 1/2
        self.z_sz = np.sqrt(np.prod(self.target_sz + context)) # ([长，宽]+2P) x 2 添加 padding  没有乘以缩放因子
        self.x_sz = self.z_sz *config.instance_sz / config.exemplar_sz  # 226   没有乘以缩放因子
        # z是初始模板的大小 x是搜索区域
        # exemplar image 
        self.avg_color = np.mean(img, axis=(0, 1)) # 计算RGB通道的均值,使用图像均值进行padding
        z = crop_and_resize(img, self.center, self.z_sz,
            out_size=config.exemplar_sz,
            border_value=self.avg_color)
        
        #对所有的图片进行预处理，得到127x127大小的patch
        # exemplar features
        z = torch.from_numpy(z).cuda().permute(2, 0, 1).unsqueeze(0).float() 

        self.kernel = self.net.features(z)
    
    @torch.no_grad()      #
    def update(self, img):
        # set to evaluation mode
        self.net.eval()

        # search images
        x = [crop_and_resize(
            img, self.center, self.x_sz * f,
            out_size=config.instance_sz,
            border_value=self.avg_color) for f in self.scale_factors]
        x = np.stack(x, axis=0)
        x = torch.from_numpy(x).cuda().permute(0, 3, 1, 2).float()
        
        # responses
        x = self.net.features(x)
        responses = self.net.head(self.kernel, x)
        responses = responses.squeeze(1).cpu().numpy()

        # upsample responses and penalize scale changes
        responses = np.stack([cv2.resize(
            u, (self.upscale_sz, self.upscale_sz),
            interpolation=cv2.INTER_CUBIC)
            for u in responses])
        responses[:config.scale_num // 2] *= config.scale_penalty
        responses[config.scale_num // 2 + 1:] *= config.scale_penalty

        # peak scale
        scale_id = np.argmax(np.amax(responses, axis=(1, 2)))

        # peak location
        response = responses[scale_id]
        response -= response.min()
        response /= response.sum() + 1e-16
        response = (1 - config.window_influence) * response + \
            config.window_influence * self.hann_window
        loc = np.unravel_index(response.argmax(), response.shape)

        # locate target center
        disp_in_response = np.array(loc) - (self.upscale_sz - 1) / 2
        disp_in_instance = disp_in_response * \
            config.total_stride / config.response_up
        disp_in_image = disp_in_instance * self.x_sz * \
            self.scale_factors[scale_id] / config.instance_sz
        self.center += disp_in_image

        # update target size
        scale =  (1 - config.scale_lr) * 1.0 + config.scale_lr * self.scale_factors[scale_id]
        self.target_sz *= scale
        self.z_sz *= scale
        self.x_sz *= scale

        # return 1-indexed and left-top based bounding box  [x,y,w,h]
        box = np.array([
            self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
            self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
            self.target_sz[1], self.target_sz[0]])

        return box
    
    def track(self, img_files, box, visualize=False):  # x,y,w,h
        frame_num = len(img_files)
        boxes = np.zeros((frame_num, 4))
        boxes[0] = box
        times = np.zeros(frame_num)

        for f, img_file in enumerate(img_files):
            img = read_image(img_file)
            begin = time.time()
            if f == 0:
                self.init(img, box)
            else:
                boxes[f, :] = self.update(img)
            times[f] = time.time() - begin

            if visualize:
                show_image(img, boxes[f, :])

        return boxes, times
    
    def train_step(self, batch, backward=True):
        # set network mode
        self.net.train(backward) # 训练模式

        # parse batch data
        z = batch[0].cuda()#to(self.device, non_blocking=self.cuda)
        x = batch[1].cuda()#to(self.device, non_blocking=self.cuda)

        with torch.set_grad_enabled(backward):
            # inference
            responses = self.net(z, x)

            # calculate loss
            labels = self._create_labels(responses.size())
            loss = self.criterion(responses, labels)
            
            if backward:
                # back propagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
        return loss.item()

    # 在禁止计算梯度下调用被允许计算梯度的函数
    @torch.enable_grad()
    def train_over(self, seqs, val_seqs=None,save_dir='models'):
        # set to train mode
        logger = get_logger('./models/logs/train_log.log')
        logger.info('start training!')
        self.net.train()
        # create save_dir folder
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # setup dataset 
        transforms = SiamFCTransforms(
                exemplar_sz=config.exemplar_sz, #127
                instance_sz=config.instance_sz, #255
                context=config.context)  # 
        
        dataset = Pair(seqs=seqs,transforms=transforms)
        
        # setup dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True)

        # loop over epochs
        for epoch in range(config.epoch_num):
            # update lr at each epoch
            self.lr_scheduler.step(epoch=epoch)

            # loop over dataloader 
            for it, batch in tqdm(enumerate(dataloader)):

                loss = self.train_step(batch, backward=True)

                #print('Epoch: {} [{}/{}] Loss: {:.5f}'.format(epoch + 1, it + 1, len(dataloader), loss))
                logger.info('Epoch: {} [{}/{}] Loss: {:.5f}'.format(epoch + 1, it + 1, len(dataloader), loss))
                
                sys.stdout.flush()
            
            # save checkpoint
            if not os.path.exists(save_dir):

                os.makedirs(save_dir)

            model_path = os.path.join(save_dir, 'siamfc_%d.pth' % (epoch + 1))

            if  torch.cuda.device_count()>1:# 多GPU

                torch.save(self.net.module.state_dict(), model_path)

            else: #单GPU
                torch.save(self.net.state_dict(), model_path)

    def _create_labels(self, size):
        # skip if same sized labels already created
        if hasattr(self, 'labels') and self.labels.size() == size:

            return self.labels

        def logistic_labels(x, y, r_pos, r_neg):

            dist = np.abs(x) + np.abs(y)  # block distance

            labels = np.where(dist <= r_pos,
                              np.ones_like(x),
                              np.where(dist < r_neg,np.ones_like(x) * 0.5,np.zeros_like(x)))

            return labels

        # distances along x- and y-axis
        n, c, h, w = size
        x = np.arange(w) - (w - 1) / 2
        y = np.arange(h) - (h - 1) / 2
        x, y = np.meshgrid(x, y)

        # create logistic labels
        r_pos = config.r_pos / config.total_stride
        r_neg = config.r_neg / config.total_stride
        labels = logistic_labels(x, y, r_pos, r_neg)

        # repeat to size
        labels = labels.reshape((1, 1, h, w))
        labels = np.tile(labels, (n, c, 1, 1))

        # convert to tensors
        self.labels = torch.from_numpy(labels).cuda().float()
        
        return self.labels
