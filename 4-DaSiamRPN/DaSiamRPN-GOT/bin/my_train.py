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

from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
from collections import OrderedDict

import setproctitle
import argparse

import sys
sys.path.append(os.getcwd())

from IPython import embed

from dasiamrpn.config import config
from dasiamrpn.network import SiamRPNNet
#from .dataset import ImagnetVIDDataset 
from got10k.datasets import  GOT10k
from dasiamrpn.dataset import GOT10kDataset
from dasiamrpn.transforms import Normalize, ToTensor, RandomStretch, RandomCrop, CenterCrop, RandomBlur, ColorAug
from dasiamrpn.loss import rpn_smoothL1, rpn_cross_entropy_balance
from dasiamrpn.visual import visual
from dasiamrpn.utils import get_topk_box, add_box_img, compute_iou, box_transform_inv, adjust_learning_rate,freeze_layers, get_logger

from IPython import embed

torch.manual_seed(config.seed)

def train(data_dir, resume_path=None, vis_port=None, init=None):

    #-----------------------
    name='GOT-10k'
    seq_dataset_train= GOT10k(data_dir, subset='train')
    seq_dataset_val = GOT10k(data_dir, subset='val')
    print('seq_dataset_train', len(seq_dataset_train))  # train-9335 个文件 
   
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

    # create dataset 
    # -----------------------------------------------------------------------------------------------------
    # train_dataset = ImagnetVIDDataset(db, train_videos, data_dir, train_z_transforms, train_x_transforms)
    train_dataset  = GOT10kDataset(
        seq_dataset_train, train_z_transforms, train_x_transforms, name)
  
    valid_dataset  = GOT10kDataset(
        seq_dataset_val, valid_z_transforms, valid_x_transforms, name)
   
    anchors = train_dataset.anchors

    # create dataloader
    
    trainloader = DataLoader(  dataset    = train_dataset,
                                batch_size = config.train_batch_size,
                                shuffle    = True, num_workers= config.train_num_workers,
                                pin_memory = True,drop_last =True)
                                
    validloader = DataLoader(dataset = valid_dataset, batch_size=config.valid_batch_size ,
                             shuffle=False, pin_memory=True,
                             num_workers=config.valid_num_workers, drop_last=True)
    
    # create summary writer
    if not os.path.exists(config.log_dir):
        os.mkdir(config.log_dir)
    summary_writer = SummaryWriter(config.log_dir)
    if vis_port:
        vis = visual(port=vis_port)

    # start training
    # -----------------------------------------------------------------------------------------------------#
    model = SiamRPNNet()

    model = model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr,momentum=config.momentum, weight_decay=config.weight_decay)
  
    #load model weight
    # -----------------------------------------------------------------------------------------------------#
    start_epoch = 1
    if resume_path and init: #不加载optimizer
        print("init training with checkpoint %s" % resume_path + '\n')
        print('------------------------------------------------------------------------------------------------ \n')
        checkpoint = torch.load(resume_path)
        if 'model' in checkpoint.keys():
            model.load_state_dict(checkpoint['model'])
        else:
            model_dict = model.state_dict()#获取网络参数
            model_dict.update(checkpoint)#更新网络参数
            model.load_state_dict(model_dict)#加载网络参数
        del checkpoint
        torch.cuda.empty_cache()#清空缓存
        print("inited checkpoint")
    elif resume_path and not init: #获取某一个checkpoint恢复训练
        print("loading checkpoint %s" % resume_path + '\n')
        print('------------------------------------------------------------------------------------------------ \n')
        checkpoint = torch.load(resume_path)
        if 'epoch' in checkpoint: 
            start_epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            model.load_state_dict(checkpoint)

        del checkpoint
        torch.cuda.empty_cache()  #缓存清零
        print("loaded checkpoint")
    elif not resume_path and config.pretrained_model: #加载预习训练模型
        print("loading pretrained model %s" % config.pretrained_model + '\n')
        print('------------------------------------------------------------------------------------------------ \n')
        checkpoint = torch.load(config.pretrained_model)
        # change name and load parameters
        checkpoint = {k.replace('features.features', 'featureExtract'): v for k, v in checkpoint.items()}
        model_dict = model.state_dict()
        model_dict.update(checkpoint)
        model.load_state_dict(model_dict)

    # print(model.featureExtract[:10])

    if torch.cuda.device_count() > 1:#如果有两块GPU，则开启多GPU模式
        model = nn.DataParallel(model)
    for epoch in range(start_epoch, config.epoch + 1):
        train_loss = []
        model.train() # 训练模式

        if config.fix_former_3_layers: #True，固定模型的前10层参数不变
            if torch.cuda.device_count() > 1: #多GPU
                freeze_layers(model.module) 
            else: # 单GPU
                freeze_layers(model)

        loss_temp_cls = 0
        loss_temp_reg = 0
        for i, data in enumerate(tqdm(trainloader)):
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
            loss = cls_loss + config.lamb * reg_loss #分类权重和回归权重
            optimizer.zero_grad()#梯度
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)#config.clip=10 ，clip_grad_norm_梯度裁剪，防止梯度爆炸
            optimizer.step()

            step = (epoch - 1) * len(trainloader) + i
            summary_writer.add_scalar('train/cls_loss', cls_loss.data, step)
            summary_writer.add_scalar('train/reg_loss', reg_loss.data, step)
            train_loss.append(loss.detach().cpu())#当前计算图中分离下来的，但是仍指向原变量的存放位置,requires_grad=false
            loss_temp_cls += cls_loss.detach().cpu().numpy()
            loss_temp_reg += reg_loss.detach().cpu().numpy()
            # if vis_port:
            #     vis.plot_error({'rpn_cls_loss': cls_loss.detach().cpu().numpy().ravel()[0],
            #                     'rpn_regress_loss': reg_loss.detach().cpu().numpy().ravel()[0]}, win=0)
            if (i + 1) % config.show_interval == 0:
            #if (i + 1) % 5 == 0:
                tqdm.write("[epoch %2d][iter %4d] cls_loss: %.4f, reg_loss: %.4f lr: %.2e"
                           % (epoch, i, loss_temp_cls / config.show_interval, loss_temp_reg / config.show_interval,
                              optimizer.param_groups[0]['lr']))
                loss_temp_cls = 0
                loss_temp_reg = 0
                # if vis_port:
                #     anchors_show = train_dataset.anchors#[1805,4]
                #     exem_img = exemplar_imgs[0].cpu().numpy().transpose(1, 2, 0)#[127,127,3]
                #     inst_img = instance_imgs[0].cpu().numpy().transpose(1, 2, 0)#ans[271,271,3] #h，w，c

                #     # show detected box with max score
                #     topk = config.show_topK# topK=3
                #     vis.plot_img(exem_img.transpose(2, 0, 1), win=1, name='exemple')
                #     cls_pred = conf_target[0]#cls_pred=[1805]   conf_target存储的是真实的标签
                #     gt_box = get_topk_box(cls_pred, regression_target[0], anchors_show)[0]#只显示第一个gt-box

                #     # show gt_box
                #     img_box = add_box_img(inst_img, gt_box, color=(255, 0, 0))
                #     vis.plot_img(img_box.transpose(2, 0, 1), win=2, name='instance')#c，h，w
                   
                #     # show anchor with max cls—score
                #     cls_pred = F.softmax(pred_conf, dim=2)[0, :, 1]
                #     scores, index = torch.topk(cls_pred, k=topk)
                #     img_box = add_box_img(inst_img, anchors_show[index.cpu()])
                #     img_box = add_box_img(img_box, gt_box, color=(255, 0, 0))
                #     vis.plot_img(img_box.transpose(2, 0, 1), win=3, name='anchor_max_score')
                #     # show pred box and gt-box 
                    
                #     cls_pred = F.softmax(pred_conf, dim=2)[0, :, 1]
                #     topk_box = get_topk_box(cls_pred, pred_offset[0], anchors_show, topk=topk)#
                #     img_box = add_box_img(inst_img, topk_box)
                #     img_box = add_box_img(img_box, gt_box, color=(255, 0, 0))
                #     vis.plot_img(img_box.transpose(2, 0, 1), win=4, name='box_max_score')

                #     # show anchor and gt-box with max iou
                #     iou = compute_iou(anchors_show, gt_box).flatten()#计算anchor和gt-box的iou
                #     index = np.argsort(iou)[-topk:]#argsort对iou元素从小到大排列，返回对应的index，并取最大的三个index
                #     img_box = add_box_img(inst_img, anchors_show[index])
                #     img_box = add_box_img(img_box, gt_box, color=(255, 0, 0))
                #     vis.plot_img(img_box.transpose(2, 0, 1), win=5, name='anchor_max_iou')

                #     # detected box
                #     regress_offset = pred_offset[0].cpu().detach().numpy()
                #     topk_offset = regress_offset[index, :]
                #     anchors_det = anchors_show[index, :]
                #     pred_box = box_transform_inv(anchors_det, topk_offset)
                #     img_box = add_box_img(inst_img, pred_box)
                #     img_box = add_box_img(img_box, gt_box, color=(255, 0, 0))
                #     vis.plot_img(img_box.transpose(2, 0, 1), win=6, name='box_max_iou')

        train_loss = np.mean(train_loss)

        valid_loss = []
        # model.eval()
        # for i, data in enumerate(tqdm(validloader)):
        #     exemplar_imgs, instance_imgs, regression_target, conf_target = data

        #     regression_target, conf_target = regression_target.cuda(), conf_target.cuda()

        #     pred_score, pred_regression = model(exemplar_imgs.cuda(), instance_imgs.cuda())

        #     pred_conf = pred_score.reshape(-1, 2, config.anchor_num * config.score_size * config.score_size).permute(0,
        #                                                                                                              2,
        #                                                                                                              1)
        #     pred_offset = pred_regression.reshape(-1, 4,
        #                                           config.anchor_num * config.score_size * config.score_size).permute(0,
        #                                                                                                              2,
        #                                                                                                              1)
        #     cls_loss = rpn_cross_entropy_balance(pred_conf, conf_target, config.num_pos, config.num_neg, anchors,
        #                                          ohem_pos=config.ohem_pos, ohem_neg=config.ohem_neg)
        #     reg_loss = rpn_smoothL1(pred_offset, regression_target, conf_target, config.num_pos, ohem=config.ohem_reg)
        #     loss = cls_loss + config.lamb * reg_loss
        #     valid_loss.append(loss.detach().cpu())
        # valid_loss = np.mean(valid_loss)
        
        
        valid_loss=0

        print("EPOCH %d valid_loss: %.4f, train_loss: %.4f" % (epoch, valid_loss, train_loss))
        
        summary_writer.add_scalar('valid/loss',valid_loss, (epoch + 1) * len(trainloader))
        
        adjust_learning_rate(optimizer,config.gamma)  # adjust before save, and it will be epoch+1's lr when next load
       
        if epoch % config.save_interval == 0:
            if not os.path.exists('./models/'):
                os.makedirs("./models/")
            save_name = "./models/dasiamrpn_{}.pth".format(epoch)
            #new_state_dict = model.state_dict()
            if torch.cuda.device_count() > 1: # 多GPU训练
                new_state_dict=model.module.state_dict()
            else:  #单GPU训练
                new_state_dict=model.state_dict()
            torch.save({
                'epoch': epoch,
                'model': new_state_dict,
                'optimizer': optimizer.state_dict(),
            }, save_name)
            print('save model: {}'.format(save_name))


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" #多卡情况下默认多卡训练,如果想单卡训练,设置为"0"

if __name__ == '__main__':
    
    # 参 数
    parser=argparse.ArgumentParser(description=" SiamRPN Train")
    parser.add_argument('--resume_path',default='./models/SiamRPNBIG.model', type=str, help="  ") # resume_path 为空, 默认加载预训练模型alexnet,在config中有配置
    parser.add_argument('--data',default='./data/GOT-10k',type=str,help=" the path of data")

    args=parser.parse_args()

    # 训 练 
    train(args.data,args.resume_path)  
    
    