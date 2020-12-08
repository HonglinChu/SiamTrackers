import numpy as np
import cv2
import torch
import torch.nn.functional as F
import time
import warnings
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
from tqdm import tqdm
from .alexnet import SiameseAlexNet
from .config import config
from .custom_transforms import ToTensor
from .utils import get_exemplar_image, get_pyramid_instance_image, get_instance_image, show_image

torch.set_num_threads(1) # otherwise pytorch will take all cpus

class SiamFCTracker:
    def __init__(self,model_path, gpu_id=0, is_deterministic=False):
        
        self.gpu_id = gpu_id
        self.name='SiamFC'
        self.is_deterministic = is_deterministic #暂时不用考虑他的用处

        with torch.cuda.device(gpu_id):
            self.model = SiameseAlexNet(gpu_id, train=False) #加载模型
            self.model.load_state_dict(torch.load(model_path))#加载模型参数
            self.model = self.model.cuda() #将模型搬到gpu上
            self.model.eval() #设置为test模式，而非train模式
        self.transforms = transforms.Compose([ToTensor()]) #在这里只要将数据转换成张量就可以了

    def _cosine_window(self, size):
        """
            get the cosine window
        """
        cos_window = np.hanning(int(size[0]))[:, np.newaxis].dot(np.hanning(int(size[1]))[np.newaxis, :])
        cos_window = cos_window.astype(np.float32)
        cos_window /= np.sum(cos_window)
        return cos_window

    def init(self, frame, box):
        """ initialize siamfc tracker
        Args:
            frame: an RGB image
            bbox: one-based bounding box [x, y, width, height]
        """
        #修改
        # self.bbox = np.array([
        #     box[1] - 1 + (box[3] - 1) / 2,
        #     box[0] - 1 + (box[2] - 1) / 2,
        #     box[3], box[2]], dtype=np.float32)
        # self.pos, self.target_sz = box[:2], box[2:] #最原始的图片大小 从groundtruth读取
        
        self.bbox = (box[0]-1, box[1]-1, box[0]-1+box[2], box[1]-1+box[3]) # zero based x1，y1,x2,y2
        self.pos = np.array([box[0]-1+(box[2])/2, box[1]-1+(box[3])/2])    # zero based cx, cy,
        self.target_sz = np.array([box[2], box[3]])                        # zero based w, h

        # get exemplar img
        self.img_mean = tuple(map(int, frame.mean(axis=(0, 1))))
        exemplar_img, scale_z, s_z = get_exemplar_image(frame, self.bbox,
                config.exemplar_size, config.context_amount, self.img_mean)

        # get exemplar feature
        exemplar_img = self.transforms(exemplar_img)[None,:,:,:] # 1,3,127,127
        with torch.cuda.device(self.gpu_id):
            exemplar_img_var = Variable(exemplar_img.cuda())
            self.model((exemplar_img_var, None))
        #penalty在update函数里面设计
        self.penalty = np.ones((config.num_scale)) * config.scale_penalty
        self.penalty[config.num_scale//2] = 1 #[0.9745, 1, 0.9745]

        # create cosine window    上采样 stride=2^4=16, 响应图的大小 17x17
        self.interp_response_sz = config.response_up_stride * config.response_sz # 272=16x17
        self.cosine_window = self._cosine_window((self.interp_response_sz, self.interp_response_sz))
        # 设计余弦窗
        # create scalse 尺度因子，0.96， 1， 1.037
        self.scales = config.scale_step ** np.arange(np.ceil(config.num_scale/2)-config.num_scale,
                np.floor(config.num_scale/2)+1)
        
        # create s_x  ？ instance是exemplar的2倍
        self.s_x = s_z + (config.instance_size-config.exemplar_size) / scale_z #s-x原始 “搜索区域”的大小；s-z是原始模板的大小

        # arbitrary scale saturation   任意尺度饱和？？？？？？
        self.min_s_x = 0.2 * self.s_x  #搜索区域下限=原始的搜索区域的1/5

        self.max_s_x = 5 * self.s_x    #搜索区域上限=原始的搜索区域的5倍

    def update(self, frame):
        """track object based on the previous frame
        Args:
            frame: an RGB image
        Returns:
            bbox: tuple of 1-based bounding box(xmin, ymin, xmax, ymax)
        """
        size_x_scales = self.s_x * self.scales #对原始的搜索区域进行多尺度的遍历
        pyramid = get_pyramid_instance_image(frame, self.pos, config.instance_size, size_x_scales, self.img_mean)
        instance_imgs = torch.cat([self.transforms(x)[None,:,:,:] for x in pyramid], dim=0) #【3，3,255,255】
        with torch.cuda.device(self.gpu_id):
            instance_imgs_var = Variable(instance_imgs.cuda())
            response_maps = self.model((None, instance_imgs_var)) # 3,1,17,17， 因为是三种尺度
            response_maps = response_maps.data.cpu().numpy().squeeze() # 3,17,17
            response_maps_up = [cv2.resize(x, (self.interp_response_sz, self.interp_response_sz), cv2.INTER_CUBIC)
             for x in response_maps] #对17x17的response进行上采样
        # get max score ，找到每一个尺度下的最大响应值
        max_score = np.array([x.max() for x in response_maps_up]) * self.penalty# penalty=【0.9745,1,0.9745】
        
        # penalty scale change
        scale_idx = max_score.argmax() #获取最大响应尺度
        response_map = response_maps_up[scale_idx] #找到最大响应尺度对应的最大响应图
        response_map -= response_map.min() #响应值减去最小值
        response_map /= response_map.sum() #归一化
        response_map = (1 - config.window_influence) * response_map + \
                config.window_influence * self.cosine_window    #添加余弦窗
        max_r, max_c = np.unravel_index(response_map.argmax(), response_map.shape) #获取索引值在response-map中的位置
        # displacement in interpolation response 插值响应中的位移，相对于中心点的位移
        disp_response_interp = np.array([max_c, max_r]) - (self.interp_response_sz-1) / 2.
        # displacement in input 真是的样本位移  response_up_stride=16；  total_stride=8
        disp_response_input = disp_response_interp * config.total_stride / config.response_up_stride
        # displacement in frame
        scale = self.scales[scale_idx]  #
        disp_response_frame = disp_response_input * (self.s_x * scale) / config.instance_size #真实的相对位移
        # position in frame coordinates
        self.pos += disp_response_frame
        # scale damping and saturation #比例阻尼和饱和度
        self.s_x *= ((1 - config.scale_lr) + config.scale_lr * scale) #尺度更新
        self.s_x = max(self.min_s_x, min(self.max_s_x, self.s_x))
        self.target_sz = ((1 - config.scale_lr) + config.scale_lr * scale) * self.target_sz #尺度更新

#         box = (self.pos[0] - self.target_sz[0]/2 + 1, # xmin   convert to 1-based
#                  self.pos[1] - self.target_sz[1]/2 + 1, # ymin
#                  self.pos[0] + self.target_sz[0]/2 + 1, # xmax
#                  self.pos[1] + self.target_sz[1]/2 + 1) # ymax
       # x,y,w,h   # top-left  w,h

        box=np.array([                                   
           self.pos[0] + 1 - (self.target_sz[0]) / 2,
           self.pos[1] + 1 - (self.target_sz[1]) / 2,
           self.target_sz[0], self.target_sz[1]])

        return box
    #获取视频序列，进行测试
    def track(self, img_files, box, visualize=False):
        frame_num = len(img_files)
        boxes = np.zeros((frame_num, 4))
        boxes[0] = box # x，y, w, h
        times = np.zeros(frame_num)

        for f, img_file in tqdm(enumerate(img_files),total=len(img_files)):
            img = cv2.imread(img_file, cv2.IMREAD_COLOR) #读取彩色图像
            begin = time.time() #开始计时
            if f == 0: #如果是第一帧
                self.init(img, box)
                #bbox = (bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]) # 1-idx
            else:
                boxes[f, :] = self.update(img)
            times[f] = time.time() - begin

            if visualize:
                show_image(img, boxes[f, :])

        return boxes, times
