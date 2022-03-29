import torch
import torch.nn as nn
import math  
from nanotrack.core.xcorr import xcorr_fast, xcorr_depthwise, xcorr_pixelwise

class BAN(nn.Module):
    def __init__(self):
        super(BAN, self).__init__()

    def forward(self, z_f, x_f):
        raise NotImplementedError

class UPChannelBAN(BAN):
    def __init__(self, feature_in=256, cls_out_channels=2):
        super(UPChannelBAN, self).__init__()

        cls_output = cls_out_channels
        loc_output = 4 

        self.template_cls_conv = nn.Conv2d(feature_in, 
                feature_in * cls_output, kernel_size=3)
        self.template_loc_conv = nn.Conv2d(feature_in, 
                feature_in * loc_output, kernel_size=3)

        self.search_cls_conv = nn.Conv2d(feature_in, 
                feature_in, kernel_size=3)
        self.search_loc_conv = nn.Conv2d(feature_in, 
                feature_in, kernel_size=3)

        self.loc_adjust = nn.Conv2d(loc_output, loc_output, kernel_size=1)


    def forward(self, z_f, x_f):
        cls_kernel = self.template_cls_conv(z_f)
        loc_kernel = self.template_loc_conv(z_f)

        cls_feature = self.search_cls_conv(x_f)
        loc_feature = self.search_loc_conv(x_f)

        cls = xcorr_fast(cls_feature, cls_kernel)
        loc = self.loc_adjust(xcorr_fast(loc_feature, loc_kernel))
        return cls, loc


class DepthwiseXCorr(nn.Module): 
    def __init__(self, in_channels,  out_channels, kernel_size=3):
        super(DepthwiseXCorr, self).__init__()

        self.conv_kernel = nn.Sequential(
                # dw 
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, groups=out_channels, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU6(inplace=True),
                # pw
                nn.Conv2d(out_channels,out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels)
                )
        
        self.conv_search = nn.Sequential(
                # dw
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, groups=in_channels,bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU6(inplace=True),
                # pw 
                nn.Conv2d(in_channels,out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels),
                ) 
        
        for modules in [self.conv_kernel, self.conv_search]:
            for m in modules.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0, 0.01)
                    m.bias.data.zero_() 

    def forward(self, kernel, search):
        kernel = self.conv_kernel(kernel) 
        search = self.conv_search(search) 
        feature = xcorr_depthwise(search, kernel)
        return feature 

class CAModule(nn.Module):
    """Channel attention module"""

    def __init__(self, channels=64, reduction=1):
        super(CAModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

class PixelwiseXCorr(nn.Module): 
    def __init__(self, in_channels,  out_channels, kernel_size=3):
        super(PixelwiseXCorr, self).__init__()

        self.CA_layer = CAModule(channels=64)

    def forward(self, kernel, search):  
       
        feature = xcorr_pixelwise(search,kernel) #

        corr = self.CA_layer(feature) 

        return corr  

class DepthwiseBAN(BAN): 
    def __init__(self, in_channels=64, out_channels=64, weighted=False):
        super(DepthwiseBAN, self).__init__()

        #self.corr_dw = DepthwiseXCorr(in_channels, out_channels)
        self.corr_pw = PixelwiseXCorr(in_channels, out_channels)
        
        cls_tower = []
        bbox_tower = [] 
        
        #------------------------------------------------------cls-----------------------------------------------------#
        # layer0
        cls_tower.append(nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2, groups=64, bias=False))
        cls_tower.append(nn.Conv2d(64,128, kernel_size=1, stride=1, padding=0, bias=False))
        cls_tower.append(nn.BatchNorm2d(128))  
        cls_tower.append(nn.ReLU6(inplace=True))   
        
        # layer1
        cls_tower.append(nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2, groups=128, bias=False))
        cls_tower.append(nn.Conv2d(128,128, kernel_size=1, stride=1, padding=0, bias=False))
        cls_tower.append(nn.BatchNorm2d(128))
        cls_tower.append(nn.ReLU6(inplace=True)) 

        # layer2
        cls_tower.append(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, groups=128, bias=False))
        cls_tower.append(nn.Conv2d(128,128, kernel_size=1, stride=1, padding=0, bias=False))
        cls_tower.append(nn.BatchNorm2d(128))
        cls_tower.append(nn.ReLU6(inplace=True)) 

        #------------------------------------------------------bbox-----------------------------------------------------#
        # layer0 
        bbox_tower.append(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, groups=64, bias=False))
        bbox_tower.append(nn.Conv2d(64,96, kernel_size=1, stride=1, padding=0, bias=False))
        bbox_tower.append(nn.BatchNorm2d(96))
        bbox_tower.append(nn.ReLU6(inplace=True)) 
       
        # layer1
        bbox_tower.append(nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1, groups=96, bias=False))
        bbox_tower.append(nn.Conv2d(96, 96, kernel_size=1, stride=1, padding=0, bias=False))
        bbox_tower.append(nn.BatchNorm2d(96))
        bbox_tower.append(nn.ReLU6(inplace=True))

        # layer2
        bbox_tower.append(nn.Conv2d(96,96, kernel_size=5, stride=1, padding=2, groups=96, bias=False))
        bbox_tower.append(nn.Conv2d(96,96, kernel_size=1, stride=1, padding=0, bias=False))
        bbox_tower.append(nn.BatchNorm2d(96))
        bbox_tower.append(nn.ReLU6(inplace=True))

        # layer3
        bbox_tower.append(nn.Conv2d(96, 96, kernel_size=5, stride=1, padding=2, groups=96, bias=False))
        bbox_tower.append(nn.Conv2d(96,96, kernel_size=1, stride=1, padding=0, bias=False))
        bbox_tower.append(nn.BatchNorm2d(96))
        bbox_tower.append(nn.ReLU6(inplace=True)) 

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))

        self.cls_logits =nn.Sequential(
            nn.Conv2d(128,  2, kernel_size=1, stride=1, padding=0),    
        ) 
        
        self.bbox_pred =nn.Sequential( 
            nn.Conv2d(96, 4, kernel_size=1, stride=1, padding=0),
        ) 

        for modules in [self.cls_tower, self.bbox_tower,
                        self.cls_logits, self.bbox_pred]:
            for m in modules.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1) 
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear): 
                    m.weight.data.normal_(0, 0.01) 
                    m.bias.data.zero_()

    def forward(self, z_f, x_f): 

        x_cls_reg = self.corr_pw(z_f, x_f) 

        cls_tower = self.cls_tower(x_cls_reg)  
        logits = self.cls_logits(cls_tower) 

        bbox_tower=self.bbox_tower(x_cls_reg) 
        bbox_reg=self.bbox_pred(bbox_tower) 
        bbox_reg = torch.exp(bbox_reg)  
        
        return logits, bbox_reg 
