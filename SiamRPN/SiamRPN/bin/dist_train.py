import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
import numpy as np
import pandas as pd
import os
import cv2
import pickle
import lmdb
import torch.nn as nn
import time

import setproctitle
import argparse

import sys
sys.path.append(os.getcwd())

from IPython import embed

from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
from collections import OrderedDict

from got10k.datasets import  GOT10k

from siamrpn.config import config
from siamrpn.network import SiamRPNNet
from siamrpn.dataset import GOT10kDataset
from siamrpn.transforms import Normalize, ToTensor, RandomStretch, RandomCrop, CenterCrop, RandomBlur, ColorAug
from siamrpn.loss import rpn_smoothL1, rpn_cross_entropy_balance
from siamrpn.visual import visual
from siamrpn.utils import get_topk_box, add_box_img, compute_iou, box_transform_inv, adjust_learning_rate,freeze_layers
from siamrpn.log_helper import *
from IPython import embed

torch.manual_seed(config.seed)

#---------------------------------dist_train-------------------------------------------#
import logging 
from torch.utils.data.distributed import DistributedSampler  #使用 DistributedSampler 对数据集进行划分,保证每一个batch的数据被分摊到每一个进程上,每个进程读取不同数据
from torch.nn.parallel import DistributedDataParallel
from siamrpn.distributed import *
#---------------------------------------------------------------------------------#

logger = logging.getLogger('global')

def dist_train(args):

    data_dir=args.data
    resume_path=args.resume_path
    vis_port=None
    init=None 

    # 是否有可用的GPU设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #---------------------------------------------初始化进程通信------------------------------------#
    # torch.cuda.set_device(args.local_rank)# 在开始训练之前为当前进程指定GPU   
    # torch.distributed.init_process_group(backend='nccl',init_method='env://') #初始化进程通信的方式
    rank, world_size = dist_init() 
    #-------------------------------------------------end-----------------------------------------#
   
    if  rank == 0:
        if not os.path.exists(config.log_dir):
            os.makedirs(config.log_dir)
        init_log('global', logging.INFO)
        add_file_handler('global',
                            os.path.join('./models/logs/logs.txt'),
                            logging.INFO)
   
    name='GOT-10k'  
    seq_dataset_train= GOT10k(data_dir, subset='train')
    #seq_dataset_val = GOT10k(data_dir, subset='val')
    logger.info('seq_dataset_train:%d'%len(seq_dataset_train))  # train-9335 个文件 
   
    # define transforms 
    train_z_transforms = transforms.Compose([
        ToTensor()
    ])
    train_x_transforms = transforms.Compose([
        ToTensor()
    ])
    valid_z_transforms = transforms.Compose([
        ToTensor()
    ])
    valid_x_transforms = transforms.Compose([
        ToTensor()
    ])

    #------------------------------------------- (2) 加载数据 DataLoader------------------------------------#
    train_dataset  = GOT10kDataset(
        seq_dataset_train, train_z_transforms, train_x_transforms, name)
    
    #(2.1)  先对所有的数据进行打乱shuffle; 把数据分配给不同的GPU设备 (这里分配的只是索引) 
    # 实例化类对象
    train_sampler = DistributedSampler(train_dataset)
   
    anchors = train_dataset.anchors
    
    # 方法1 单进程,单GPU
    # trainloader = DataLoader(dataset = train_dataset,
    #                          batch_size = config.train_batch_size,
    #                          shuffle    = True, num_workers= config.train_num_workers,
    #                          pin_memory = True, drop_last =True)
    
    # 方法2 多进程多GPU
    trainloader = DataLoader(   dataset    = train_dataset,
                                sampler    = train_sampler,
                                batch_size = config.train_batch_size,
                                shuffle    = False, num_workers= config.train_num_workers,
                                pin_memory = True,drop_last =True)   
    # 方法3 多进程多GPU
    # train_batch_sampler =torch.utils.data.BatchSampler(train_sampler, config.train_batch_size, drop_last=True)
    # trainloader = DataLoader(   dataset    = train_dataset,
    #                             batch_sampler =train_batch_sampler,
    #                             shuffle    = False, num_workers= config.train_num_workers,
    #                             pin_memory = True,drop_last =True,sampler=train_sampler)    
    #-------------------------------------------end----------------------------------------------------------#
    
    if rank==0: # 全局的rank
        summary_writer = SummaryWriter(config.log_dir)
    else:
        summary_writer = None 
    
    #-------------------------------------------------------(3) 模型的初始化--------------------------------------------#
    # DistributedDataParallel帮助为不同的GPU上求得到梯度进行all reduce(汇总不同GPU计算所得的梯度, 并同步计算结果),
    # all reduce 之后不同GPU中模型的梯度均为 all reduce 之前各GPU梯度的均值
    model = SiamRPNNet()
    model = model.cuda()# 一定要先load进GPU 
    #model.to(device)

    start_epoch = 1

    #(1)加载checkpoint,不加载optimizer参数
    # 注意 torch.load的时候指定map_location, 否则会导致第一块GPU占用更多的资源
    if resume_path and init: 
        logger.info("init training with checkpoint %s" % resume_path)
        checkpoint = torch.load(resume_path,map_location=device)
        if 'model' in checkpoint.keys():
            model.load_state_dict(checkpoint['model'])
        else:
            model_dict = model.state_dict()#获取网络参数
            model_dict.update(checkpoint)#更新网络参数
            model.load_state_dict(model_dict)#加载网络参数
        del checkpoint 
        torch.cuda.empty_cache()#清空缓存
        logger.info("inited checkpoint")

    #(2)获取某一个checkpoint恢复训练,加载optimizer参数
    elif resume_path and not init: 
        logger.info("loading checkpoint %s" % resume_path)
        checkpoint = torch.load(resume_path,map_location=device)
        if 'model' in checkpoint.keys():
            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint) #加载模型的时候可以设置 model.load_state_dict(checkpoint,strict=False) 只对关键字相同的参数进行赋值
        del checkpoint
        torch.cuda.empty_cache()  #缓存清零
        logger.info("loaded checkpoint")

    #(3)加载预习训练模型
    elif not resume_path and config.pretrained_model: 
        logger.info("loading pretrained model %s" % config.pretrained_model)
        checkpoint = torch.load(config.pretrained_model,map_location=device) # 这里要指定map_location
        checkpoint = {k.replace('features.features', 'featureExtract'): v for k, v in checkpoint.items()}

        model_dict = model.state_dict()

        model_dict.update(checkpoint)

        model.load_state_dict(model_dict) #

        del checkpoint

        torch.cuda.empty_cache()  #缓存清零 
    
    # DDP 包装模型
    model = DistributedDataParallel(model,find_unused_parameters=True, device_ids=[args.local_rank]) 

    # device_ids需要是唯一的一个GPU编号，这个训练脚本在这个GPU上执行,device_ids需要是[args.local_rank]并且output_device需要是args.local_rank才能使用这个工具
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr,momentum=config.momentum, weight_decay=config.weight_decay)

    logger.info('start training!')

    for epoch in range(start_epoch, config.epoch + 1):

        # DistributedSampler 类中有set_epoch方法
        train_sampler.set_epoch(epoch) #数据打乱 

        model.train() # 训练模式

        if config.fix_former_3_layers: #True，固定模型的前10层参数不变
            if torch.cuda.device_count() > 1: #多GPU
                freeze_layers(model.module) 
            else: # 单GPU
                freeze_layers(model)
        train_loss = []
        loss_temp_cls = 0
        loss_temp_reg = 0

        # 在0GPU上显示进度条
        if args.local_rank==0:
            
            trainloader=tqdm(trainloader)

        for i, data in enumerate(trainloader):

            exemplar_imgs, instance_imgs, regression_target, conf_target = data
            # conf_target (8,1125) (8,225x5)
            regression_target, conf_target = regression_target.cuda(), conf_target.cuda()
            #pre_score=64,10,19,19 ； pre_regression=[64,20,19,19]
            pred_score, pred_regression = model(exemplar_imgs.cuda(), instance_imgs.cuda())
            # [64, 5x19x19, 2]=[64,1805,2]
            pred_conf = pred_score.reshape(-1, 2, config.anchor_num * config.score_size * config.score_size).permute(0,
                                                                                                                     2,
                                                                                                                     1)
             #[64,5x19x19,4] =[64,1805,4]               
            pred_offset = pred_regression.reshape(-1, 4,
                                                  config.anchor_num * config.score_size * config.score_size).permute(0,
                                                                                                                     2,
                                                                                                                     1)
            cls_loss = rpn_cross_entropy_balance(pred_conf, conf_target, config.num_pos, config.num_neg, anchors,
                                                 ohem_pos=config.ohem_pos, ohem_neg=config.ohem_neg)
            reg_loss = rpn_smoothL1(pred_offset, regression_target, conf_target, config.num_pos, ohem=config.ohem_reg)
            loss = cls_loss + config.lamb * reg_loss 

            #loss=average_reduce(loss)

            optimizer.zero_grad()
            loss.backward()

            # 这一步非常重要
            reduce_gradients(model)# 对多个GPU的梯度求均值

            # reduce_value()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip) 
            optimizer.step()
            
            step = (epoch - 1) * len(trainloader) + i

            if  get_rank() ==0:
                summary_writer.add_scalar('train/cls_loss', cls_loss.data, step)
                summary_writer.add_scalar('train/reg_loss', reg_loss.data, step)

            train_loss.append(average_reduce(loss).detach().cpu()) # 注意这里的average_reduce返回的是tensor类型
            # loss_temp_cls.append(average_reduce(cls_loss.detach().cpu()))
            # loss_temp_reg.append(average_reduce(reg_loss.detach().cpu()))

            loss_temp_cls += average_reduce(cls_loss).detach().cpu()
            loss_temp_reg += average_reduce(reg_loss).detach().cpu()

            if (i + 1) % config.show_interval == 0:

                logger.info('Epoch:[{}/{}]  cls_loss={:.4f}  reg_loss={:.4f} lr={:.2e}'.format(epoch, config.epoch, loss_temp_cls / config.show_interval,loss_temp_reg / config.show_interval,optimizer.param_groups[0]['lr']))

                loss_temp_cls = 0
                loss_temp_reg = 0
        #torch.mean()必须输入tensor,最后得到的结果也是tensor,torch.stack()沿着某个维度进行堆叠数据
        #torch.mean(torch.stack(train_loss)) # train_loss是tensor,则必须使用这个,否则报错
        ##np.mean()输入list可以，array也可以
        train_loss = np.mean(train_loss)

        valid_loss = []

        valid_loss=0

        #print("EPOCH %d valid_loss: %.4f, train_loss: %.4f" % (epoch, valid_loss, train_loss))
        logger.info('Epoch:[{}/{}]  valid_loss={:.4f}  train_loss={:.4f}'.format(epoch, config.epoch, valid_loss, train_loss))
        
        if  get_rank() ==0:
            summary_writer.add_scalar('valid/loss',valid_loss, (epoch + 1) * len(trainloader))
        
        adjust_learning_rate(optimizer,config.gamma)  # adjust before save, and it will be epoch+1's lr when next load
       
        if epoch % config.save_interval == 0:
            if not os.path.exists('./models/'):
                os.makedirs("./models/")
            save_name = "./models/siamrpn_{}.pth".format(epoch)
            #new_state_dict = model.state_dict()
            if torch.cuda.device_count() > 1: # 多GPU训练
                new_state_dict=model.module.state_dict()
            else:  #单GPU训练
                new_state_dict=model.state_dict()
            if get_rank()==0:
                torch.save({
                    'epoch': epoch,
                    'model': new_state_dict,
                    'optimizer': optimizer.state_dict(),
                }, save_name)
            logger.info('save model: {}'.format(save_name))
        
        # 训练完成之后,释放进程组资源
        dist.destroy_process_group() 

import random 
def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    
    seed_torch(6666)
    
    # 参数
    parser=argparse.ArgumentParser(description=" SiamRPN Train")  
    # 在训练的train.py中必须要解析--local_rank=LOCAL_PROCESS_RANK这个命令行参数，这个命令行参数是由torch.distributed.launch提供的，指定了每个GPU在本地的rank
    # local_rank是一个局部的id，在每个机器上GPU的id
    parser.add_argument('--local_rank',type=int, default=0,help='Node rank for distributed training')
    # resume_path 为空,默认加载alexnet模型
    parser.add_argument('--resume_path',default='', type=str, help=" input gpu id ") # resume_path 为空, 默认加载预训练模型alexnet,在config中有配置

    parser.add_argument('--data',default='./data/GOT-10k',type=str,help=" the path of data") 

    args=parser.parse_args()
   
    dist_train(args) 

    