import numpy as np
import cv2
import torch
import torch.nn.functional as F
import time
import torchvision.transforms as transforms
from tqdm import tqdm
from .network import SiameseAlexNet
from .config import config
from .custom_transforms import ToTensor
from .utils import get_exemplar_image, get_instance_image, box_transform_inv,add_box_img,add_box_img_left_top,show_image
from .generate_anchors import generate_anchors

from IPython import embed

torch.set_num_threads(1)  # otherwise pytorch will take all cpus

class SiamRPNTracker:
    def __init__(self, model_path,gpu_id,is_deterministic=False):

        self.gpu_id=gpu_id
        self.name='SiamRPN'
        self.model = SiameseAlexNet()#这里不是加载模型的么?为何被注释掉了
        self.is_deterministic = is_deterministic

        checkpoint = torch.load(model_path)

        if 'model' in checkpoint.keys():
            self.model.load_state_dict(torch.load(model_path)['model'])#这里好像是多GPU并行的操作
        else:
            self.model.load_state_dict(torch.load(model_path))
        with torch.cuda.device(self.gpu_id):
            self.model = self.model.cuda()
        self.model.eval()
        self.transforms = transforms.Compose([ToTensor()])

        valid_scope = 2 * config.valid_scope + 1 # 2x9+1=19 or 2x8+1=17 ； 2x7+1=15
        self.anchors = generate_anchors(config.total_stride, config.anchor_base_size, config.anchor_scales,
                                        config.anchor_ratios,
                                        valid_scope)
        self.window = np.tile(np.outer(np.hanning(config.score_size), np.hanning(config.score_size))[None, :],
                              [config.anchor_num, 1, 1]).flatten()

    def _cosine_window(self, size):
        """
            get the cosine window
        """
        cos_window = np.hanning(int(size[0]))[:, np.newaxis].dot(np.hanning(int(size[1]))[np.newaxis, :])
        cos_window = cos_window.astype(np.float32)
        cos_window /= np.sum(cos_window)
        return cos_window


    def init(self, frame, bbox):
        """ initialize siamrpn tracker
        Args:
            frame: an RGB image
            bbox: one-based bounding box [x, y, width, height]
        """
        #self.bbox = np.array([bbox[0] + bbox[2] / 2 - 1 / 2, bbox[1] + bbox[3] / 2 - 1 / 2, bbox[2], bbox[3]]) #cx,cy,w,h
        self.bbox = np.array([bbox[0]-1 + (bbox[2]-1) / 2 , bbox[1]-1 + (bbox[3]-1) / 2 , bbox[2], bbox[3]]) #cx,cy,w,h
        
        self.pos = np.array([bbox[0]-1 + (bbox[2]-1) / 2 , bbox[1]-1 + (bbox[3]-1) / 2])  # center x, center y, zero based
        
        self.target_sz = np.array([bbox[2], bbox[3]])  # width, height
       
        self.origin_target_sz = np.array([bbox[2], bbox[3]])# w,h

        # get exemplar img
        self.img_mean = np.mean(frame, axis=(0, 1))
        #获取模板图像
        exemplar_img, scale_z, _ = get_exemplar_image(frame, self.bbox,config.exemplar_size, config.context_amount, self.img_mean)
        # img = add_box_img_left_top(frame,self.bbox)
        # gt_box = np.array([0,0,scale_z*self.bbox[2],scale_z*self.bbox[3]])
        # img = add_box_img(exemplar_img,gt_box)
        # get exemplar feature
        exemplar_img = self.transforms(exemplar_img)[None, :, :, :]#在测试阶段，转换成tensor类型就可以了
        self.model.track_init(exemplar_img.cuda(self.gpu_id))

    def update(self, frame):
        """track object based on the previous frame
        Args:
            frame: an RGB image

        Returns:
            bbox: tuple of 1-based bounding box(xmin, ymin, xmax, ymax)
        """
        instance_img_np, _, _, scale_x = get_instance_image(frame, self.bbox, config.exemplar_size,
                                                         config.instance_size,
                                                         config.context_amount, self.img_mean)
        
        instance_img = self.transforms(instance_img_np)[None, :, :, :]
        # pred_score=1,2x5,17,17 ; pre_regression=1,4x5,17,17 
        pred_score, pred_regression = self.model.track(instance_img.cuda(self.gpu_id))# 
        #[1,5x17x17,2] 5x17x17=1445
        pred_conf = pred_score.reshape(-1, 2, config.anchor_num * config.score_size * config.score_size).permute(0,2,1)
        #[1,5x17x17,4]                                                                                                     
        pred_offset = pred_regression.reshape(-1, 4,config.anchor_num * config.score_size * config.score_size).permute(0,2,1)
                                                                                                                
        delta = pred_offset[0].cpu().detach().numpy()
        #使用detach()函数来切断一些分支的反向传播;返回一个新的Variable，从当前计算图中分离下来的，但是仍指向原变量的存放位置,不同之处只是requires_grad为false，得到的这个Variable永远不需要计算其梯度，不具有grad。
        #即使之后重新将它的requires_grad置为true,它也不会具有梯度grad #这样我们就会继续使用这个新的Variable进行计算，后面当我们进行反向传播时，到该调用detach()的Variable就会停止，不能再继续向前进行传播
        
        box_pred = box_transform_inv(self.anchors, delta) #通过 anchors 和 offset 来预测box
        #pred_conf=[1,1805,2]; 
        #hah=F.softmax(pred_conf, dim=2)
        score_pred = F.softmax(pred_conf, dim=2)[0, :, 1].cpu().detach().numpy()#计算预测分类得分
        #?
        def change(r): 
            return np.maximum(r, 1. / r) # x 和 y 逐位进行比较选择最大值
        #?
        def sz(w, h):
            pad = (w + h) * 0.5
            sz2 = (w + pad) * (h + pad)
            return np.sqrt(sz2)
        #?
        def sz_wh(wh):
            pad = (wh[0] + wh[1]) * 0.5
            sz2 = (wh[0] + pad) * (wh[1] + pad)
            return np.sqrt(sz2)

        s_c = change(sz(box_pred[:, 2], box_pred[:, 3]) / (sz_wh(self.target_sz * scale_x)))  # scale penalty
        r_c = change((self.target_sz[0] / self.target_sz[1]) / (box_pred[:, 2] / box_pred[:, 3]))  # ratio penalty
        penalty = np.exp(-(r_c * s_c - 1.) * config.penalty_k)#尺度惩罚和比例惩罚
        pscore = penalty * score_pred#对每一个anchors的分类预测×惩罚因子
        pscore = pscore * (1 - config.window_influence) + self.window * config.window_influence #再乘以余弦窗
        best_pscore_id = np.argmax(pscore) #得到最大的得分
        
        target = box_pred[best_pscore_id, :] / scale_x #target（x,y,w,h）是以上一帧的pos为（0,0）

        lr = penalty[best_pscore_id] * score_pred[best_pscore_id] * config.lr_box#预测框的学习率
        #关于clip的用法，如果
        res_x = np.clip(target[0] + self.pos[0], 0, frame.shape[1])#w=frame.shape[1]
        res_y = np.clip(target[1] + self.pos[1], 0, frame.shape[0])#h=frame.shape[0]

        res_w = np.clip(self.target_sz[0] * (1 - lr) + target[2] * lr, config.min_scale * self.origin_target_sz[0],
                        config.max_scale * self.origin_target_sz[0])
        res_h = np.clip(self.target_sz[1] * (1 - lr) + target[3] * lr, config.min_scale * self.origin_target_sz[1],
                        config.max_scale * self.origin_target_sz[1])

        self.pos = np.array([res_x, res_y]) #更新之后的坐标

        self.target_sz = np.array([res_w, res_h])

        bbox = np.array([res_x, res_y, res_w, res_h])

        self.bbox = ( #cx, cy, w, h
            np.clip(bbox[0], 0, frame.shape[1]).astype(np.float64),
            np.clip(bbox[1], 0, frame.shape[0]).astype(np.float64),
            np.clip(bbox[2], 10, frame.shape[1]).astype(np.float64),
            np.clip(bbox[3], 10, frame.shape[0]).astype(np.float64)) 

        #这个用来画图使用
        bbox=np.array([# tr-x,tr-y w,h                                  
            self.pos[0] + 1 - (self.target_sz[0]-1) / 2,
            self.pos[1] + 1 - (self.target_sz[1]-1) / 2,
            self.target_sz[0], self.target_sz[1]])

        #return self.bbox, score_pred[best_pscore_id]
        return bbox

    #数据集测试
    def track(self, img_files, box, visualize=False):
        frame_num = len(img_files)
        boxes = np.zeros((frame_num, 4))
        boxes[0] = box
        times = np.zeros(frame_num)

        for f, img_file in tqdm(enumerate(img_files),total=len(img_files)):
            img = cv2.imread(img_file, cv2.IMREAD_COLOR)
            begin = time.time()
            if f == 0:
                self.init(img, box)
                #bbox = (bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]) # 1-idx
            else:
                boxes[f, :] = self.update(img)
            times[f] = time.time() - begin

            if visualize:
                show_image(img, boxes[f, :])

        return boxes, times
