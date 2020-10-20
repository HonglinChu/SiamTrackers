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

from fire import Fire
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter

from .config import config
from .alexnet import SiameseAlexNet
from .dataset import ImagnetVIDDataset
from .custom_transforms import Normalize, ToTensor, RandomStretch, RandomCrop, CenterCrop, RandomBlur, ColorAug

torch.manual_seed(1234)

def train(gpu_id, data_dir):

    # loading meta data
    meta_data_path = os.path.join(data_dir, "meta_data.pkl") #
    # pickle文件将python项目过程中用到的一些暂时变量，或者需要提取， 暂存的字符串，列表，字典等数据保存起来
    # 如果加载保存过的pickle文件，可以立刻复原之前程序运行中的对象
    # pickle 提供四个功能: dumps, dump, loads, load
    meta_data = pickle.load(open(meta_data_path,'rb')) # rb是打开二进制文件，文本文件用r打开
    all_videos = [x[0] for x in meta_data] # meta_data 包括图片名字和视频序列，all_videos包括了所有视频序列的名字
    
    # split train/valid dataset rain_test_split (train_data-被划分的样本特征；train-target:被划分的样本标签；test_size-测试结合样本所占的比例/数量；random_state:随机数的种子)
    train_videos, valid_videos = train_test_split(all_videos, test_size=1-config.train_ratio, random_state=config.seed)

    # define transforms  
    random_crop_size = config.instance_size - 2 * config.total_stride  #  ？本身图像经过 Curation 之后是 255x255 的大小，现在要随机裁剪（255-2*8）x（255-2*8）
    train_z_transforms = transforms.Compose([  #  train z 是初始样本的大小，exemplar
        RandomStretch(),
        CenterCrop((config.exemplar_size, config.exemplar_size)), # 仅仅从 255x255 图像的中间区域裁剪 127x127 大小，
        ToTensor()
    ])
    train_x_transforms = transforms.Compose([  #  train x 代表搜索区域的大小 instance
        RandomStretch(),
        RandomCrop((random_crop_size, random_crop_size), # 239x239 
                    config.max_translate),               # max translation of random shift
        ToTensor() # 转换成 Tensor
    ])
    valid_z_transforms = transforms.Compose([
        CenterCrop((config.exemplar_size, config.exemplar_size)),  # valid 没有 RandomStretch 
        ToTensor()
    ]) #
    valid_x_transforms = transforms.Compose([ToTensor()])

    # lmdb 是一款开源的高效快速的内存映射数据库，C语言编写，基于B+树索引。他不是一个需要独立运行的数据库管理进程，只要在需要访问lmdb数据库的代码里面引用lmdb库，给出数据库所在的目录，就能方便地实现读写lmdb数据库
    # open lmdb  如果在对应路径文件夹下没有data.mbd或lock.mdb文件，则会生成一个空的，如果有，不会覆盖 ，map_size用来定义最大存储容量，单位是kb
    
    db = lmdb.open(data_dir+'.lmdb', readonly=True, map_size=int(50e9))
    
    # create dataset 

    train_dataset = ImagnetVIDDataset(db, train_videos, data_dir,train_z_transforms, train_x_transforms)
    #train_dataset = ImagnetVIDDataset(train_videos, data_dir,train_z_transforms, train_x_transforms)

    valid_dataset = ImagnetVIDDataset(db, valid_videos, data_dir,valid_z_transforms, valid_x_transforms, training=False)
    #valid_dataset = ImagnetVIDDataset(valid_videos, data_dir,valid_z_transforms, valid_x_transforms, training=False)

    # create dataloader  drop_last=True 数据集长度除以余下的数据被抛弃
    trainloader = DataLoader(train_dataset, batch_size=config.train_batch_size,
            shuffle=True, pin_memory=True,  num_workers= config.train_num_workers, drop_last=True) #

    validloader = DataLoader(valid_dataset, batch_size=config.valid_batch_size,
            shuffle=False, pin_memory=True, num_workers= config.valid_num_workers, drop_last=True)#
    
    # create summary writer tensorboardX
    if not os.path.exists(config.log_dir):
        os.mkdir(config.log_dir)

    summary_writer = SummaryWriter(config.log_dir) # 

    # start training
    with torch.cuda.device(gpu_id):
        model = SiameseAlexNet(gpu_id, train=True) #__init__
        model.init_weights() #init_weights  主要是卷积层和归一化层参数初始化
        '''
        在cuda脚本中 GPU的指定方式
        （1）
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        （2）
        import torch
        torch.cuda.set_device(id)
        （3）
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        '''
        #model=torch.nn.DataParallel(model,device_ids=[0,1])# 多GPU并行计算
        model = model.cuda() #单个GPU计算，gpu_id
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr,
                momentum=config.momentum, weight_decay=config.weight_decay) #没有正则项，没有动量
        #动态调整学习率
        scheduler = StepLR(optimizer, step_size=config.step_size, gamma=config.gamma) 

        for epoch in range(config.epoch): #epoch=30
            train_loss = []
            model.train() #设置模型为训练模式
            for i, data in enumerate(tqdm(trainloader)):
            #for i, data in trainloader:
                exemplar_imgs, instance_imgs = data # 单线程模式，每次调用都会返回一对
                exemplar_var, instance_var = Variable(exemplar_imgs.cuda()), Variable(instance_imgs.cuda()) #这两个必须是Variable吗
                optimizer.zero_grad() #梯度清零
                outputs = model((exemplar_var, instance_var)) #[8, 1, 15, 15]
                loss = model.weighted_loss(outputs) #loss是tensor
                loss.backward()
                optimizer.step()#梯度参数更新
                step = epoch * len(trainloader) + i
                summary_writer.add_scalar('train/loss', loss.data, step)
                train_loss.append(loss.data)
            #train_loss = np.mean(train_loss) #train_loss是一个list，而loss是一个tensor，所以
            train_loss = torch.mean(torch.stack(train_loss))
            valid_loss = []
            model.eval() #test模式
            for i, data in enumerate(tqdm(validloader)):
                exemplar_imgs, instance_imgs = data
                exemplar_var, instance_var = Variable(exemplar_imgs.cuda()),Variable(instance_imgs.cuda())

                outputs = model((exemplar_var, instance_var))
                loss = model.weighted_loss(outputs)
                valid_loss.append(loss.data)
            #valid_loss = np.mean(valid_loss)
            valid_loss = torch.mean(torch.stack(valid_loss))

            print("EPOCH %d valid_loss: %.4f, train_loss: %.4f" %(epoch, valid_loss, train_loss))

            summary_writer.add_scalar('valid/loss', valid_loss, (epoch+1)*len(trainloader))

            torch.save(model.cpu().state_dict(), "./models/siamfc_{}.pth".format(epoch+1)) #把model转换到cpu上进行保存

            model.cuda() #把模型再次转换到gpu上
            scheduler.step()#调整学习率
