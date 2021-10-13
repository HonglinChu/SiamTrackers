import argparse
import shutil
from os.path import join, isdir, isfile
from os import makedirs

#from dataset import VID
import time
import pdb
import torch
import numpy as np
import torch.nn as nn
from net_upd import UpdateResNet512,UpdateResNet256
from torch.utils.data import dataloader
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

#from scipy import io

parser = argparse.ArgumentParser(description='Training  in Pytorch 0.4.0')
parser.add_argument('--input_sz', dest='input_sz', default=125, type=int, help='crop input size')
parser.add_argument('--padding', dest='padding', default=2.0, type=float, help='crop padding size')
parser.add_argument('--range', dest='range', default=10, type=int, help='select range')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 32)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-5, type=float,
                    metavar='W', help='weight decay (default: 5e-5)')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--save', '-s', default='./updatenet/checkpoint', type=str, help='directory for saving')

args = parser.parse_args()

print(args)
best_loss = 1e6

dataram = dict()
 
#训练数据路径
#tem_path ='./updatenet/update_set_templates_step1_std'
tem_path ='./updatenet/update_set_templates_step2_std'

#特征图
dataram['template0'] = np.load(join(tem_path,'template0.npy'))
dataram['template']  = np.load(join(tem_path,'template.npy'))
dataram['templatei'] = np.load(join(tem_path,'templatei.npy'))

#索引号
dataram['pre'] = np.load(join(tem_path,'pre.npy'))#累积特征index
dataram['gt'] = np.load(join(tem_path,'gt.npy'))# gt特征index
dataram['init0'] = np.load(join(tem_path,'init0.npy'))#初始化模板index
dataram['train'] = np.arange(len(dataram['gt']), dtype=np.int)#生成0到len(50)之间的随机数字

# optionally resume from a checkpoint
if args.resume:
    if isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

cudnn.benchmark = True

save_path = args.save

def adjust_learning_rate(optimizer, epoch, lr0):
    #lr = np.logspace(-lr0[0], -lr0[1], num=args.epochs)[epoch] #构造一个从10的负lr0[0]次方到10的lr0[1]次方的等比数列，数列长度是50
    lrs=np.logspace(-7, -8, num=50) #每一个epoch的学习率都是一样的
    lr = lrs[epoch]
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr 

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, epoch,lr, filename=join(save_path, 'checkpoint.pth.tar')):
    # name0 = 'lr' + str(lr[0])+'_'+str(lr[1])
    # name0 = name0.replace('.','_') #
    # epo_path = join(save_path, name0)
    # if not isdir(epo_path):
    #     makedirs(epo_path)
    if epoch<50:
        if (epoch+1) % 1 == 0:
            filename=join(save_path, 'checkpoint{}.pth.tar'.format(epoch+1))
            torch.save(state, filename)    

#lrs = np.array([[4, 6],[4, 7],[4.5, 5],[4.5, 6],[4.5, 7],[5, 5],[5, 6],[5, 7],[5, 8],[6, 6],[6, 7],[6, 8],[7, 7],[7, 8],[6.5, 6.5],[6.5, 7],[6.5, 8],[7, 9],[7, 10],[8, 8],[8, 9],[9, 9],[9, 10],[10, 10]])

lrs=np.array([[6,7],[7,8]]) #重新定义学习率 step=[10e-6 , 10e-7]  step2=[10e-7, 10e-8]

for ii in np.arange(0,1):

    # construct model
    model = UpdateResNet512()#512通道
    #updatenet = UpdateResNet256() 
    
    #加载预训练模型
    model.load_state_dict(torch.load('./models/vot2018.pth.tar')['state_dict'])
    
    model = nn.DataParallel(model,device_ids=[0,1])#开启多GPU
    model.cuda()

    #update_model_fix = dict()
    #for i in update_model.keys():
    #    update_model_fix['.'.join(i.split('.')[1:])] = update_model[i]
    #updatenet.load_state_dict(update_model_fix)
    
    criterion = nn.MSELoss(size_average=False).cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr, #lr=0.01,实际上并没有发挥作用
                            momentum=args.momentum,          #momentum=0.9
                            weight_decay=args.weight_decay)  #weight_decay=5e-5

    for epoch in range(args.start_epoch, args.epochs): # 0-epochs=50

        #adjust_learning_rate(optimizer, epoch, lrs[ii])#默认每一个epoch的学习率是固定的10-7
        adjust_learning_rate(optimizer, epoch, lrs[ii]) #默认每一个epoch的学习率是固定的10-7

        losses = AverageMeter()
        #subset = shuffle(subset)    
        subset = np.random.permutation(dataram['train'])#对序列进行随机排序 permutation不在原数组上操作，shuffle直接在原数组操作
        for t in range(0, len(subset), args.batch_size):#start：0； stop：len(subset)=45578；  step:batch_size=64
            
            batchStart = t #  batchStart和 batchEnd相隔64
            batchEnd = min(t+args.batch_size, len(subset))
            batch = subset[batchStart:batchEnd]#获取一个batch
            init_index = dataram['init0'][batch]
            pre_index = dataram['pre'][batch]
            gt_index = dataram['gt'][batch]
            
            # reset diff T0 为何重新选择距离初始模板的偏差（是因为可以避免选择到最后一个？？）
            for rr in range(len(init_index)):
                if init_index[rr] != 0:#不是初始帧
                    init_index[rr] = np.random.choice(init_index[rr],1)#numpy.random.choice(a, size=None, replace=True, p=None)

            cur = dataram['templatei'][batch]             #当前帧的检测模板
            init = dataram['template0'][batch-init_index] #初始模板
            pre = dataram['template'][batch-pre_index]    #累积模板
            gt = dataram['template0'][batch+gt_index-1]# gt_index一般为1, gt取当前帧和下一帧是差不多的
            
            #pdb.set_trace() 
            temp = np.concatenate((init, pre, cur), axis=1) #[64, 512, 6, 6]-->[64,1536,6,6]
            input_up = torch.Tensor(temp) #输入
            target = torch.Tensor(gt)
            init_inp = Variable(torch.Tensor(init)).cuda()#
            input_up = Variable(input_up).cuda()
            target = Variable(target).cuda()
            
            # compute output
            output = model(input_up, init_inp)#[64,512,6,6]
            loss = criterion(output, target)/target.size(0) #target.size(0)=64; 

            # measure accuracy and record loss
            loss_data=loss.cpu().data.numpy().tolist()           
            losses.update(loss_data)#

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if t % args.print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                       str(epoch).zfill(2), str(t).zfill(5), len(subset), loss=losses))     
        save_checkpoint({'state_dict': model.state_dict()}, epoch,lrs[ii])        
