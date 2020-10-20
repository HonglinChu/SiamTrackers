import os
import sys
sys.path.append(os.getcwd())
import argparse
from fire import Fire
from siamfc import train

#在命令行模式运行的时候不能开启一下命令
#import multiprocessing
#multiprocessing.set_start_method('spawn',True)

gpu_id=0

data_dir='./data/ILSVRC_VID_CURATION'

# model_gpu=nn.DataParallel(model,device_ids=[0,1])# 多GPU并行计算

# output=model_gpu(input)

if __name__ == '__main__':

    # Fire(train) # Command Line Interfaces  生成命令行接口
   
    # 参数
    parser=argparse.ArgumentParser(description=" SiamFC Train")
    parser.add_argument('--gpu',default=gpu_id, type=int, help=" input gpu id ")
    parser.add_argument('--data',default=data_dir,type=str,help=" the path of data")
    args=parser.parse_args()

    # 训练
    train(args.gpu, args.data)




