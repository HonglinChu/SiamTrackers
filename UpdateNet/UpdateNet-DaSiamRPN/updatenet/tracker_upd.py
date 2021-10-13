# --------------------------------------------------------
# DaSiamRPN
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from utils_upd import get_axis_aligned_bbox, get_subwindow_tracking, Round, generate_anchor
from config_upd import Config as TrackerConfig

def tracker_eval(net, x_crop, target_pos, target_sz, window, scale_z, p):
    delta, score = net(x_crop)
    
    delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1).data.cpu().numpy()
    score = F.softmax(score.permute(1, 2, 3, 0).contiguous().view(2, -1), dim=0).data[1, :].cpu().numpy()

    delta[0, :] = delta[0, :] * p.anchor[:, 2] + p.anchor[:, 0]
    delta[1, :] = delta[1, :] * p.anchor[:, 3] + p.anchor[:, 1]
    delta[2, :] = np.exp(delta[2, :]) * p.anchor[:, 2]
    delta[3, :] = np.exp(delta[3, :]) * p.anchor[:, 3]

    def change(r):
        return np.maximum(r, 1./r)

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


def SiamRPN_init_upd(im, target_pos, target_sz, net):
# def SiamRPN_init_upd(im, init_rbox, net):

    #[cx, cy, w, h] = get_axis_aligned_bbox(init_rbox)
        # tracker init
    #target_pos, target_sz = np.array([cx, cy]), np.array([w, h])

    state = dict()
    p = TrackerConfig()
    p.update(net.cfg)
    #p.update(net.cfg)
    state['im_h'] = im.shape[0]
    state['im_w'] = im.shape[1]
    if p.adaptive:
        if ((target_sz[0] * target_sz[1]) / float(state['im_h'] * state['im_w'])) < 0.004:
            p.instance_size = 287  # small object big search region
        else:
            p.instance_size = 271
    
    #python2
    #p.score_size = (p.instance_size - p.exemplar_size) // p.total_stride + 1 #271-127/8+1=19
    #python3
    p.score_size = (p.instance_size - p.exemplar_size) // p.total_stride + 1 #271-127/8+1=19

    p.anchor = generate_anchor(p.total_stride, p.scales, p.ratios, int(p.score_size))

    avg_chans = np.mean(im, axis=(0, 1))

    wc_z = target_sz[0] + p.context_amount * sum(target_sz)
    hc_z = target_sz[1] + p.context_amount * sum(target_sz)
   
    #s_z = round(np.sqrt(wc_z * hc_z))#python2
    s_z =  np.sqrt(wc_z * hc_z)#python3和python2Round

    # initialize the exemplar
    z_crop = get_subwindow_tracking(im, target_pos, p.exemplar_size, Round(s_z), avg_chans)#Round是python3里面的
    
    z = Variable(z_crop.unsqueeze(0))
    z_f = net.featextract(z.cuda()) #[1,512,6,6]
    #net.temple(z.cuda())
    net.kernel(z_f)

    if p.windowing == 'cosine':
        window = np.outer(np.hanning(p.score_size), np.hanning(p.score_size))
    elif p.windowing == 'uniform':
        window = np.ones((p.score_size, p.score_size))
    window = np.tile(window.flatten(), p.anchor_num)

    state['p'] = p
    state['net'] = net
    state['avg_chans'] = avg_chans
    state['window'] = window
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    state['z_f_cur'] = z_f.cpu().data#当前检测特征图
    state['z_f'] = z_f.cpu().data    #累积的特征图
    state['z_0'] = z_f.cpu().data    #初始特征图
    state['gt_f_cur']=z_f.cpu().data  #gt框对应的特征图
    return state

def SiamRPN_track_upd(state, im,updatenet):
    p = state['p']
    net = state['net']
    avg_chans = state['avg_chans']
    window = state['window']
    
    #（1）生成gt特征图
    #------------------------------------------------------start------------------------------------------------#
    
    gt_pos = state['gt_pos']
    gt_sz = state['gt_sz']

    wc_z = gt_sz[1] + p.context_amount * sum(gt_sz)
    hc_z = gt_sz[0] + p.context_amount * sum(gt_sz)
    s_z  = np.sqrt(wc_z * hc_z)#2020-05-13
    gt_crop = Variable(get_subwindow_tracking(im, gt_pos, p.exemplar_size, Round(s_z), avg_chans).unsqueeze(0))    
    g_f = net.featextract(gt_crop.cuda())
#
    #-------------------------------------------------------end---------------------------------------------------#

    #（2）生成预测特征图
    #------------------------------------------------------start-----------------------------------------------#

    target_pos = state['target_pos']
    target_sz = state['target_sz']

    wc_z = target_sz[1] + p.context_amount * sum(target_sz)
    hc_z = target_sz[0] + p.context_amount * sum(target_sz)
    s_z = np.sqrt(wc_z * hc_z)

    scale_z = p.exemplar_size / s_z
    #d_search = (p.instance_size - p.exemplar_size) / 2
    d_search = (p.instance_size - p.exemplar_size) // 2 # python3

    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    # extract scaled crops for search region x at previous target position
    x_crop = Variable(get_subwindow_tracking(im, target_pos, p.instance_size, Round(s_x), avg_chans).unsqueeze(0))

    target_pos, target_sz, score = tracker_eval(net, x_crop.cuda(), target_pos, target_sz * scale_z, window, scale_z, p)
    target_pos[0] = max(0, min(state['im_w'], target_pos[0]))
    target_pos[1] = max(0, min(state['im_h'], target_pos[1]))
    target_sz[0] = max(10, min(state['im_w'], target_sz[0]))
    target_sz[1] = max(10, min(state['im_h'], target_sz[1]))
    
    z_crop = Variable(get_subwindow_tracking(im, target_pos, p.exemplar_size, Round(s_z), avg_chans).unsqueeze(0))    
   
    z_f = net.featextract(z_crop.cuda())#当前检测模板

    # 模板更新方式1-Linear
    if updatenet=='':
        zLR=0.0102 #SiamFC默认的更新频率
        z_f_ = (1-zLR) * Variable(state['z_f']).cuda() + zLR * z_f # 累积模板
    # temp = np.concatenate((init, pre, cur), axis=1)

    # 模板更新方式2-UpdateNet
    else:
        temp = torch.cat((Variable(state['z_0']).cuda(),Variable(state['z_f']).cuda(),z_f),1)
        init_inp = Variable(state['z_0']).cuda()
        z_f_ = updatenet(temp,init_inp)#累积特征图

    net.kernel(z_f_)
    
    state['target_pos'] = target_pos
    state['target_sz'] = target_sz
    state['score'] = score
    state['z_f'] = z_f_.cpu().data   #累积模板
    state['z_f_cur']=z_f.cpu().data  #当前检测模板
    state['gt_f_cur']=g_f.cpu().data #当前帧gt框对应的特征模板
    state['net'] = net
    return state
