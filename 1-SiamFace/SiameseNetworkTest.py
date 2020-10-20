# # One Shot Learning with Siamese Networks
import torchvision
import torchvision.datasets as dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from SiameseUtil import *
import pandas as pd


# 加载
net = SiameseNetwork()
net.load_state_dict(torch.load("best.siamese.ph"))
net.eval()

# ## 测试
# 有 3 个人的图像没有参与训练, 我们使用他们做测试. 两个图像的距离越小, 他们越有可能属于同一个人
folder_dataset_test = dataset.ImageFolder(root=Config.testing_dir)
siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
                                        transform=transforms.Compose([transforms.Resize((100,100)),
                                                                      transforms.ToTensor()
                                                                      ]),
                                        keepOrder=True)

test_dataloader = DataLoader(siamese_dataset,num_workers=0,batch_size=1,shuffle=False)
dataiter = iter(test_dataloader)
x0,_,_,fn,_ = next(dataiter)
df = pd.DataFrame(columns=['GroudTruth', 'TestFace', 'Similarity'])

i = -1
for data in dataiter:
    i = i + 1
    _,x1,label2, fn1, fn2 = data 
    concatenated = torch.cat((x0,x1),0)
    
    output1,output2 = net(Variable(x0),Variable(x1))
    euclidean_distance = F.pairwise_distance(output1, output2)
    similarity = 1 - euclidean_distance.item()
    imshow(torchvision.utils.make_grid(concatenated),'Similarity: {:.2f}'.format(similarity))
    df.loc[i] = [fn, fn1, similarity]

df = df.sort_values(by=['Similarity'],ascending=False)
export_csv = df.to_csv ('results.csv', index = None, header=True) 
print (df)
