import os
import sys
import setproctitle
import argparse
from fire import Fire

sys.path.append(os.getcwd())
from siamrpn.train import train
from IPython import embed

#在命令行模式下运行的时候不能开启一下命令
# import multiprocessing
# multiprocessing.set_start_method('spawn',True)

#def train(data_dir, model_path=None, vis_port=None, init=None):
#python bin/train_siamrpn.py --data_dir /PATH/TO/SAVE_DATA
data_dir='./data/ytb_vid_curation'
model_path='./models/siamrpn_38.pth'

if __name__ == '__main__':
    #program_name = 'zrq train ' + os.getcwd().split('/')[-1] #获 取 当 前 的 工 作 路 径
    #setproctitle.setproctitle(program_name) #将进程名字修改为 program_name
    #Fire(train)
    # 参数
    parser=argparse.ArgumentParser(description=" SiamRPN Train")
    #parser.add_argument('--gpu',default=gpu_id, type=int, help=" input gpu id ")
    parser.add_argument('--data',default=data_dir,type=str,help=" the path of data")
    args=parser.parse_args()

    # 训练
    #train(args.gpu, args.data)

    train(args.data,model_path) #

    #train(args.data) #
