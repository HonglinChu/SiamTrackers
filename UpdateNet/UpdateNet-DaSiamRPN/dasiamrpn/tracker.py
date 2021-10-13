# --------------------------------------------------------
# DaSiamRPN
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F

from .utils import get_subwindow_tracking,generate_anchor,get_axis_aligned_bbox,Round
from .net import SiamRPNBIG
from .config import TrackerConfig
from updatenet.net_upd import UpdateResNet512,UpdateResNet256

class SiamRPNTracker:

    def __init__(self, model_path,update_path,gpu_id,step=1):
        
        self.gpu_id=gpu_id
        self.net = SiamRPNBIG()#
        #self.model=SiamRPNOTB()
        #self.model=SiamRPNVOT()
        self.is_deterministic = False#???

        checkpoint = torch.load(model_path)
        if 'model' in checkpoint.keys():
            self.net.load_state_dict(torch.load(model_path)['model'])#
        else:
            self.net.load_state_dict(torch.load(model_path))
        with torch.cuda.device(self.gpu_id):
            self.net = self.net.cuda()
        self.net.eval()

        self.state = dict()

        self.step=step# 1,2,3
        if self.step==1:
            self.name='DaSiamRPN'
        elif self.step==2:
            self.name='Linear'
        else:
            # update_path='./updatenet/models/checkpoint26.pth.tar'
            dataset=update_path.split('/')[-1].split('.')[0]
            if dataset=='vot2018' or dataset=='vot2016':
                self.name='UpdateNet'
            else:
                self.name=dataset

        if self.step==3:
            #load UpdateNet network
            self.updatenet = UpdateResNet512()
            #self.updatenet = UpdateResNet256()    
            update_model=torch.load(update_path)['state_dict']

            update_model_fix = dict()
            for i in update_model.keys():
                if i.split('.')[0]=='module': #多GPU模型去掉开头的'module'
                    update_model_fix['.'.join(i.split('.')[1:])] = update_model[i]
                else:
                    update_model_fix[i]=update_model[i] #单GPU模型直接赋值

            self.updatenet.load_state_dict(update_model_fix)
                
            #self.updatenet.load_state_dict(update_model)
            self.updatenet.eval().cuda()
        else:
            self.updatenet=''
    
    def tracker_eval(self, x_crop, target_pos, target_sz, window, scale_z, p):
        delta, score = self.net(x_crop)

        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1).data.cpu().numpy()
        score = F.softmax(score.permute(1, 2, 3, 0).contiguous().view(2, -1), dim=0).data[1, :].cpu().numpy()

        delta[0, :] = delta[0, :] * p.anchor[:, 2] + p.anchor[:, 0]
        delta[1, :] = delta[1, :] * p.anchor[:, 3] + p.anchor[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * p.anchor[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * p.anchor[:, 3]

        def change(r):
           
            return np.maximum(r,1./r)

        def sz(w, h):
            pad = (w + h) * 0.5
            sz2 = (w + pad) * (h + pad)
            return np.sqrt(sz2)

        def sz_wh(wh):
            pad = (wh[0] + wh[1]) * 0.5
            sz2 = (wh[0] + pad) * (wh[1] + pad)
            return np.sqrt(sz2)

        # size penalty
        s_c = change(sz(delta[2, :], delta[3, :]) / (sz_wh(target_sz)))  # scale penalty
        r_c = change((target_sz[0] / target_sz[1]) / (delta[2, :] / delta[3, :]))  # ratio penalty
        

        penalty = np.exp(-(r_c * s_c - 1.) * p.penalty_k)
        pscore = penalty * score

        # window float
        pscore = pscore * (1 - p.window_influence) + window * p.window_influence
        best_pscore_id = np.argmax(pscore)

        target = delta[:, best_pscore_id] / scale_z
        target_sz = target_sz / scale_z
        lr = penalty[best_pscore_id] * score[best_pscore_id] * p.lr

        res_x = target[0] + target_pos[0]
        res_y = target[1] + target_pos[1]

        res_w = target_sz[0] * (1 - lr) + target[2] * lr
        res_h = target_sz[1] * (1 - lr) + target[3] * lr

        target_pos = np.array([res_x, res_y])
        target_sz = np.array([res_w, res_h])
        return target_pos, target_sz, score[best_pscore_id]


    def init(self, im, init_rbox):
         
        state=self.state

        [cx, cy, w, h] = get_axis_aligned_bbox(init_rbox)
        # tracker init
        target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
       
        p = TrackerConfig
        p.update(self.net.cfg)

        state['im_h'] = im.shape[0]
        state['im_w'] = im.shape[1]

        if p.adaptive:
            if ((target_sz[0] * target_sz[1]) / float(state['im_h'] * state['im_w'])) < 0.004:
                p.instance_size = 287  # small object big search region
            else:
                p.instance_size = 271
            #python3 
            p.score_size = (p.instance_size - p.exemplar_size) // p.total_stride + 1 #python3

        p.anchor = generate_anchor(p.total_stride, p.scales, p.ratios, p.score_size)

        avg_chans = np.mean(im, axis=(0, 1))

        wc_z = target_sz[0] + p.context_amount * sum(target_sz)
        hc_z = target_sz[1] + p.context_amount * sum(target_sz)
        s_z = Round(np.sqrt(wc_z * hc_z))#python3

        # initialize the exemplar
        z_crop = get_subwindow_tracking(im, target_pos, p.exemplar_size, s_z, avg_chans)
        z_crop = Variable(z_crop.unsqueeze(0))

        if self.step==1:#不更新模板
            self.net.temple(z_crop.cuda())#初始化模板
        else:           #更新模板
            z_f = self.net.featextract(z_crop.cuda()) #[1,512,6,6]
            self.net.kernel(z_f)
            state['z_f'] = z_f.cpu().data    #累积的模板
            state['z_0'] = z_f.cpu().data    #初始的模板

        if p.windowing == 'cosine':
            window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))
        elif p.windowing == 'uniform':
            window = np.ones((p.score_size, p.score_size))
        window = np.tile(window.flatten(), p.anchor_num)

        state['p'] = p
        state['net'] = self.net
        state['avg_chans'] = avg_chans
        state['window'] = window
        state['target_pos'] = target_pos
        state['target_sz'] = target_sz
        self.state=state

        return state

    def update(self, im):

        state=self.state
        p = state['p']
        net = state['net']
        avg_chans = state['avg_chans']
        window = state['window']
        target_pos = state['target_pos']
        target_sz = state['target_sz']

        wc_z = target_sz[1] + p.context_amount * sum(target_sz)
        hc_z = target_sz[0] + p.context_amount * sum(target_sz)
        s_z = np.sqrt(wc_z * hc_z)  # 

        scale_z = p.exemplar_size / s_z
        d_search = (p.instance_size - p.exemplar_size) / 2 #python3 2020-05-13
        pad = d_search / scale_z
        s_x = s_z + 2 * pad

        # extract scaled crops for search region x at previous target position
        #x_crop = Variable(get_subwindow_tracking(im, target_pos, p.instance_size, round(s_x), avg_chans).unsqueeze(0))
        x_c = get_subwindow_tracking(im, target_pos, p.instance_size, Round(s_x), avg_chans)
        x_crop=Variable(x_c.unsqueeze(0))
        
        target_pos, target_sz, score = self.tracker_eval( x_crop.cuda(), target_pos, target_sz * scale_z, window, scale_z, p)
        target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
        target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
        target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
        target_sz[1] = max(10, min(state['im_h'], target_sz[1]))

        #更新模板
        if  self.step>1:
            z_crop = Variable(get_subwindow_tracking(im, target_pos, p.exemplar_size, Round(s_z), avg_chans).unsqueeze(0))    
            z_f = self.net.featextract(z_crop.cuda())#检测模板
            if self.step==2:#模板更新方式1-Linear
                zLR=0.0102   #SiamFC[0.01, 0.05],  0.0102是siamfc初始化的方法
                z_f_ = (1-zLR) * Variable(state['z_f']).cuda() + zLR * z_f # 累积模板
                #temp = np.concatenate((init, pre, cur), axis=1)
            else:           #模板更新方式2-UpdateNet
                temp = torch.cat((Variable(state['z_0']).cuda(),Variable(state['z_f']).cuda(),z_f),1)
                init_inp = Variable(state['z_0']).cuda()
                z_f_ = self.updatenet(temp,init_inp)

            state['z_f'] = z_f_.cpu().data #累积模板
            self.net.kernel(z_f_)          #更新模板  
        
        state['target_pos'] = target_pos
        state['target_sz'] = target_sz
        state['score'] = score
        self.state=state
        return state
