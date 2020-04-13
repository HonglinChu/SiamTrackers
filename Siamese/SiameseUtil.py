import torchvision
import torchvision.datasets as dataset
from torch.utils.data import DataLoader,Dataset
import torch.nn as nn
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps   
import matplotlib.pyplot as plt
import torchvision.utils

# ## 帮助函数

def imshow(img,text=None,should_save=False):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()    

def show_plot(iteration,loss):
    plt.plot(iteration,loss)
    plt.show()

# ## 用于配置的帮助类
class Config():
    training_dir = "./data/faces/training/"
    testing_dir = "./data/faces/testing/"
    train_batch_size = 32    #64
    train_number_epochs = 50  # 100

# ## 定制 Dataset 类
# 这个类用于产生一对图片. 

class SiameseNetworkDataset(Dataset):
    #输入三个参数
    def __init__(self,imageFolderDataset,transform=None, keepOrder=False):
        self.imageFolderDataset = imageFolderDataset #路径   
        self.transform = transform #是否进行变换
        self.keepOrder = keepOrder #是否保持固定的顺序
    
    #重载函数，得到一组数据（x,label）
    def __getitem__(self,index): 
        if self.keepOrder:
            img_tuple = self.imageFolderDataset.imgs[index]
            img = Image.open(img_tuple[0]) # PIL Image的方法
            # 图像转为黑白灰度图
            img = img.convert("L")
            if self.transform is not None:
                img = self.transform(img)
            #返回两个一样的图片
            return img, img, torch.from_numpy(np.array([int(0)],dtype=np.float32)), img_tuple[0], img_tuple[0]

        img0_tuple = random.choice(self.imageFolderDataset.imgs)#从列表中随机选择一个图像出来
        # 使得50%的训练数据为一对图像属于同一类别
        should_get_same_class = random.randint(0,1) # 0-1随机数
        if should_get_same_class:#如果数值=1
            while True:
                # 循环直到一对图像属于同一类别
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1]==img1_tuple[1]:#如果选择的图片是同意个类别则跳出，[1]存储的是类别信息
                    break
        else: #should_get_same_class=0
            while True:
                # 循环直到一对图像属于不同的类别
        
                img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                if img0_tuple[1] !=img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0]) # [0]存储的是图片路径信息，[1]存储的是类别信息
        img1 = Image.open(img1_tuple[0])
        # 图像转为黑白灰度图
        img0 = img0.convert("L")
        img1 = img1.convert("L")

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        
        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])],dtype=np.float32)), img0_tuple[0], img1_tuple[0]
    
    def __len__(self):
        return len(self.imageFolderDataset.imgs)

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            # TODO 添加代码构成网络的卷积层
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            nn.Dropout2d(p=.2),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Dropout2d(p=.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Dropout2d(p=.2),
        )

        self.fc1 = nn.Sequential(
            # TODO 添加代码构成网络的全连接层
             nn.Linear(8*100*100, 500),
             nn.ReLU(inplace=True),
 
             nn.Linear(500, 500),
             nn.ReLU(inplace=True),
 
             nn.Linear(500, 5)
            )

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


