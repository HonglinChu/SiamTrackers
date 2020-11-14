import lmdb
import cv2
import numpy as np
import os
import hashlib
import functools
import  argparse
from glob import glob
from fire import Fire
from tqdm import tqdm
from multiprocessing import Pool
from IPython import embed
import multiprocessing as mp
import multiprocessing
multiprocessing.set_start_method('spawn',True)

def worker(video_name):
    image_names = glob(video_name + '/*')
    kv = {}
    for image_name in image_names:
        img = cv2.imread(image_name)
        _, img_encode = cv2.imencode('.jpg', img)
        img_encode = img_encode.tobytes()
        kv[hashlib.md5(image_name.encode()).digest()] = img_encode
    return kv

def create_lmdb(data_dir, output_dir, num_threads=mp.cpu_count()):
    video_names = glob(data_dir + '/*')
    video_names = [x for x in video_names if 'meta_data.pkl' not in x]
    # video_names = [x for x in video_names if os.path.isdir(x)]
    db = lmdb.open(output_dir, map_size=int(200e9))
    with Pool(processes=num_threads) as pool:
        for ret in tqdm(pool.imap_unordered(functools.partial(worker), video_names), total=len(video_names)):
            with db.begin(write=True) as txn:
                for k, v in ret.items():
                    txn.put(k, v)

data_directory='./data/ytb_vid_curation'
output_directory='./data/ytb_vid_curation.lmdb'
num_thread=2 #开12个线程速度很慢
if __name__ == '__main__':
    #Fire(create_lmdb)
    #参数
    parser=argparse.ArgumentParser(description=" Create lmdb ")
    #parser.add_argument('--gpu',default=gpu_inid, type=int, help=" input gpu id ")
    parser.add_argument('--data_dir',default=data_directory,type=str,help=" the path of data")
    parser.add_argument('--output_dir',default=output_directory,type=str,help=" the output path of data")
    parser.add_argument('--num_thread',default=num_thread,type=str,help=" the num of thread")
    args=parser.parse_args()
    create_lmdb(args.data_dir,args.output_dir,args.num_thread)
