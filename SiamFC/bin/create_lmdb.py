import lmdb
import cv2
import numpy as np
import os 
import hashlib
import functools
import argparse
import multiprocessing
from glob import glob
from fire import Fire
from tqdm import tqdm
from multiprocessing import Pool
multiprocessing.set_start_method('spawn',True)

def worker(video_name):
    image_names = glob(video_name+'/*')#读取所有的图片
    kv = {}
    for image_name in image_names:
        img = cv2.imread(image_name)
        _, img_encode = cv2.imencode('.jpg', img) #重新编码图片，编码成.jpg格式
        img_encode = img_encode.tobytes()   #从图片到字节流
        # hash.digest() 返回摘要，作为一个二进制数据字符串值 hash.hexdigest() 返回摘要，作为十六进制数据字符串值
        # hashlib.md5获取一个md5加密算法对象，digest()加密后的结果用二进制表示
        # hashlib.md5(image_name.encode()).digest() --> 图片的路径进行编码转换成二进制的键值 --> kv[key]=value
        kv[hashlib.md5(image_name.encode()).digest()] = img_encode #？？？？？？
    return kv

def create_lmdb(data_dir, output_dir, num_threads):

    video_names = glob(data_dir+'/*')

    video_names = [x for x in video_names if os.path.isdir(x)] #获取Annotation Imageset和Annotation三个文件夹
   
    db = lmdb.open(output_dir, map_size=int(50e9)) #创建名字为ILSVRC_VID_CURATION.lmdb 的文件夹，里面包含两个子文件data.mdb lock.mdb
    
    #多线程操作
    with Pool(processes=num_threads) as pool:
        for ret in tqdm(pool.imap_unordered(functools.partial(worker), video_names), total=len(video_names)):
            with db.begin(write=True) as txn:
                for k, v in ret.items():
                    txn.put(k, v)

    # 单线程操作
    # for video in video_names:
    #     ret=worker(video)
    #     #with是一个控制流语句，跟if while for try 类似，with可以用来简化try-finally代码
    #     #先执行expression，然后执行该表达式返回的对象实例的__enter__函数，然后将该函数的返回值赋给as后面的变量
    #     #注意，是将__enter__函数的返回值赋给变量
    #     with db.begin(write=True) as txn: 
    #         for k,v in ret.items():
    #             txn.put(k,v)

Data_dir='./data/ILSVRC_VID_CURATION'
Output_dir='./data/ILSVRC_VID_CURATION.lmdb'
Num_threads=8# 原来设置为32
if __name__ == '__main__':
   # Fire(create_lmdb)
    # parse arguments
    parser=argparse.ArgumentParser(description="Demo SiamFC")
    parser.add_argument('--d',default=Data_dir,type=str,help="data_dir")
    parser.add_argument('--o',default=Output_dir,type=str, help="out put")
    parser.add_argument('--n',default=Num_threads, type=int, help="thread_num")
    args=parser.parse_args()

    create_lmdb(args.d, args.o, args.n)



