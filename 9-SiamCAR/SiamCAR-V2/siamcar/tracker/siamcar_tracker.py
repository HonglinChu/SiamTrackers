# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch.nn.functional as F
import cv2
import torch  
from siamcar.core.config import cfg
from siamcar.tracker.base_tracker import SiameseTracker
from siamcar.utils.misc import bbox_clip

class SiamCARTracker(SiameseTracker):
    def __init__(self, model, cfg):
        super(SiamCARTracker, self).__init__()
        hanning = np.hanning(cfg.SCORE_SIZE)
        self.window = np.outer(hanning, hanning) # 生成 汉宁窗
        self.model = model # 跟踪网络
        self.model.eval()  # 测试模式

    def _convert_cls(self, cls):
        cls = F.softmax(cls[:,:,:,:], dim=1).data[:,1,:,:].cpu().numpy()
        return cls

    @torch.no_grad()
    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """  
        self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,  #[cx,cy]
                                    bbox[1]+(bbox[3]-1)/2])
        self.size = np.array([bbox[2], bbox[3]]) # [w, h]

        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))
        
        # calculate channle average  img.shape=[w, h, c]  
        self.channel_average = np.mean(img, axis=(0, 1))# [x,y,z]每个通道的均值

        # get crop  输入：原始图像，目标中心，期望输出图像大小，实际裁剪区域大小，通道均值填充, 输出:[1, 3, 127, 127], cuda, float32
        z_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)
        self.model.template(z_crop)

    def change(self,r):
        return np.maximum(r, 1. / r)

    def sz(self, w, h):
        pad = (w + h) * 0.5 
        return np.sqrt((w + pad) * (h + pad))

    def cal_penalty(self, ltrbs, penalty_lk):

        #l+r [25, 25]  预测的是 ltrb=[0-l, 1-t, 2-r, 3-b]  
        bboxes_w = ltrbs[0, :, :] + ltrbs[2, :, :]    
        #t+b [25, 25]
        bboxes_h = ltrbs[1, :, :] + ltrbs[3, :, :]    

        a=self.sz(bboxes_w, bboxes_h) 

        b=self.sz(self.size[0]*self.scale_z, self.size[1]*self.scale_z)

        s_c = self.change(a/b)

        r_c = self.change((self.size[0] / self.size[1]) / (bboxes_w / bboxes_h))

        penalty = np.exp(-(r_c * s_c - 1) * penalty_lk) 

        return penalty

    def accurate_location(self, max_r_up, max_c_up):
        dist = int((cfg.TRACK.INSTANCE_SIZE - (cfg.TRACK.SCORE_SIZE - 1) * 8) / 2) #？？？？193不是和255一一对应的么，怎么感觉是从255中扣出来的。
        max_r_up += dist
        max_c_up += dist
        p_cool_s = np.array([max_r_up, max_c_up])
        disp = p_cool_s - (np.array([cfg.TRACK.INSTANCE_SIZE, cfg.TRACK.INSTANCE_SIZE]) - 1.) / 2.
        return disp #求距离中心点的偏移量
    # 上采样找到最大值，然后再下采样是否精准？？
    def coarse_location(self, hp_score_up, p_score_up, scale_score, ltrbs):
        upsize = (cfg.TRACK.SCORE_SIZE - 1) * cfg.TRACK.STRIDE + 1
        max_r_up_hp, max_c_up_hp = np.unravel_index(hp_score_up.argmax(), hp_score_up.shape) # [rpw, col]
        max_r = int(round(max_r_up_hp / scale_score))
        max_c = int(round(max_c_up_hp / scale_score))
        # max_r = bbox_clip(max_r, 0, cfg.TRACK.SCORE_SIZE)# [0,25]超出范围则截断处理
        # max_c = bbox_clip(max_c, 0, cfg.TRACK.SCORE_SIZE)# [0,25]超出范围则截断处理

        max_r = bbox_clip(max_r, 0, cfg.TRACK.SCORE_SIZE-1)# 下标从0开始，这里最大值取不到SCORE_SIZE所以应该SCORE_SIZE-1，原始开源代码并没有-1操作，也没有报错，为何？
        max_c = bbox_clip(max_c, 0, cfg.TRACK.SCORE_SIZE-1)# 下标从0开始，这里最大值取不到SCORE_SIZE所以应该SCORE_SIZE-1

        bbox_region = ltrbs[max_r, max_c, :] #找到最大值对应的回归分支的坐标 

        min_bbox = int(cfg.TRACK.REGION_S * cfg.TRACK.EXEMPLAR_SIZE) # 127*0.1=12
        max_bbox = int(cfg.TRACK.REGION_L * cfg.TRACK.EXEMPLAR_SIZE) # 127*0.44=55
        l_region = int(min(max_c_up_hp, bbox_clip(bbox_region[0], min_bbox, max_bbox)) / 2.0) #？？
        t_region = int(min(max_r_up_hp, bbox_clip(bbox_region[1], min_bbox, max_bbox)) / 2.0) #？？

        r_region = int(min(upsize - max_c_up_hp, bbox_clip(bbox_region[2], min_bbox, max_bbox)) / 2.0)
        b_region = int(min(upsize - max_r_up_hp, bbox_clip(bbox_region[3], min_bbox, max_bbox)) / 2.0)
        # 注意这里的掩码操作 
        mask = np.zeros_like(p_score_up)
        mask[max_r_up_hp - t_region:max_r_up_hp + b_region + 1, max_c_up_hp - l_region:max_c_up_hp + r_region + 1] = 1
        p_score_up = p_score_up * mask 

        return p_score_up 

    def getCenter(self,hp_score_up, p_score_up, scale_score,ltrbs):
        # 我认为这一步操作可以去掉，最大值大概率会出现在mask区域，所有有没有mask对于最大值位置的确定并不影响，这里mask的计算反而增加了许多额外的计算，还有优化的空间
        # corse location  根据 hp_score_up来获取最大值的位置，以及对应ltrbs； 最有作用到 p_score_up
        score_up = self.coarse_location(hp_score_up, p_score_up, scale_score, ltrbs) # mask掩码
        # accurate location 
        max_r_up, max_c_up = np.unravel_index(score_up.argmax(), score_up.shape) # 使用mask进行初步筛选
        #max_r_up, max_c_up = np.unravel_index(p_score_up.argmax(), p_score_up.shape)，不使用mask筛选的效果
        
        disp = self.accurate_location(max_r_up,max_c_up)# 255特征图上的偏移

        disp_ori = disp / self.scale_z # 原始图上的偏移
        new_cx = disp_ori[1] + self.center_pos[0]# 
        new_cy = disp_ori[0] + self.center_pos[1]
        return max_r_up, max_c_up, new_cx, new_cy
    
    @torch.no_grad()
    def track(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        self.scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)
        # 【1， 3， 255， 255】
        outputs = self.model.track(x_crop) # [cls, loc, cen] --> [[1, 2, 25,25], [1, 4, 25,25], [1,1,25,25]]
        cls = self._convert_cls(outputs['cls']).squeeze() # [25, 25]  .data.cpu().numpy()
        cen = outputs['cen'].data.cpu().numpy()  # [1,1,25,25]
        cen = (cen - cen.min()) / cen.ptp() #x.ptp()对所有数据求最大值和最小值的差值，归一化到[0,1]之间
        cen = cen.squeeze()  #[25, 25]
        ltrbs = outputs['loc'].data.cpu().numpy().squeeze() #[1, 4, 25, 25] --> [4, 25, 25]

        upsize = (cfg.TRACK.SCORE_SIZE-1) * cfg.TRACK.STRIDE + 1  #（25-1）× 8 +1 = 193 ？？？
        penalty = self.cal_penalty(ltrbs, cfg.TRACK.PENALTY_K)
        p_score = penalty * cls * cen        

        if cfg.TRACK.hanming:
            hp_score = p_score*(1 -  cfg.TRACK.WINDOW_INFLUENCE) + self.window *  cfg.TRACK.WINDOW_INFLUENCE
        else:
            hp_score = p_score
        #25 25
        hp_score_up = cv2.resize(hp_score, (upsize, upsize), interpolation=cv2.INTER_CUBIC) # 　线性插值　[193, 193]
        p_score_up = cv2.resize(p_score, (upsize, upsize), interpolation=cv2.INTER_CUBIC) # 线性插值　[193, 193]
        cls_up = cv2.resize(cls, (upsize, upsize), interpolation=cv2.INTER_CUBIC) # 线性插值　[193, 193]
        ltrbs = np.transpose(ltrbs,(1,2,0))# [4, 25, 25] --> [25,25,4]
        ltrbs_up = cv2.resize(ltrbs, (upsize, upsize), interpolation=cv2.INTER_CUBIC) #线性插值 [25, 25, 4] --> [193, 193,4]

        scale_score = upsize / cfg.TRACK.SCORE_SIZE  # SCORE_SIZE=25
        # get center
        max_r_up, max_c_up, new_cx, new_cy = self.getCenter(hp_score_up, p_score_up, scale_score, ltrbs)
        # get w h
        ave_w = (ltrbs_up[max_r_up,max_c_up,0] + ltrbs_up[max_r_up,max_c_up,2]) / self.scale_z
        ave_h = (ltrbs_up[max_r_up,max_c_up,1] + ltrbs_up[max_r_up,max_c_up,3]) / self.scale_z
        
        # 计算惩罚权重
        s_c = self.change(self.sz(ave_w, ave_h) / self.sz(self.size[0]*self.scale_z, self.size[1]*self.scale_z))
        r_c = self.change((self.size[0] / self.size[1]) / (ave_w / ave_h))
        penalty = np.exp(-(r_c * s_c - 1) *cfg.TRACK.PENALTY_K)
        
        # 惩罚因子×分类得分×学习率=新的学习率
        lr = penalty * cls_up[max_r_up, max_c_up] * cfg.TRACK.LR # 
        new_width = lr*ave_w + (1-lr)*self.size[0] #对宽和高进行不同程度的调节 这里宽和高的学习率是否可以改成动态的？？
        new_height = lr*ave_h + (1-lr)*self.size[1]

        # clip boundary
        cx = bbox_clip(new_cx,0,img.shape[1])
        cy = bbox_clip(new_cy,0,img.shape[0])
        width = bbox_clip(new_width,0,img.shape[1])
        height = bbox_clip(new_height,0,img.shape[0])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])
        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]

        return {
                'bbox': bbox,
               }
