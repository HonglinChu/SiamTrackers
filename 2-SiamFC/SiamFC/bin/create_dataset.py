import numpy as np
import pickle
import os
import cv2
import functools
import xml.etree.ElementTree as ET
import sys
import argparse
import multiprocessing
from multiprocessing import Pool
from fire import Fire
from tqdm import tqdm
from glob import glob
from siamfc import config, get_instance_image

sys.path.append(os.getcwd())

multiprocessing.set_start_method('spawn',True) #开启多线程

def worker(output_dir, video_dir):

    image_names = glob(os.path.join(video_dir, '*.JPEG'))

    #sort函数
    #sorted()作用于任意可以迭代的对象，而sort()一般作用于列表；
    #sort()函数不需要复制原有列表，消耗的内存较少，效率也较高： b=sorted(a)并不改变a的排序，a.sort() 会改变a的排序
    #sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)  #key依据某一列为排序依据
    image_names = sorted(image_names,key=lambda x:int(x.split('/')[-1].split('.')[0])) #从小到大进行排列

    video_name = video_dir.split('/')[-1]

    save_folder = os.path.join(output_dir, video_name)

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    trajs = {}

    for image_name in image_names:
        img = cv2.imread(image_name)
        #axis=0，表示shape第0个元素被压缩成1，即求每一列的平均值，axis=1,表示输出矩阵是1列（shape第一个元素被压缩成1），求每一行的均值，axis=(0,1)表示shape的第0个元素和第1个元素被压缩成了1
        #元组和列表类似，不同之处在于元组的元素不能修改，元组使用小括号，列表使用方括号，元组的创建很简单，只需要在括号中添加元素，并使用逗号间隔开
        #map(int, img.mean(axis=(0, 1)))将数据全部转换为int类型列表
        img_mean = tuple(map(int, img.mean(axis=(0, 1))))
        anno_name = image_name.replace('Data', 'Annotations') #str.replace('a','b')将str中的a替换为字符串中的b
        anno_name = anno_name.replace('JPEG', 'xml')
        tree = ET.parse(anno_name) #解析xml文件
        root = tree.getroot() #获取根节点； 作为一个元素，root有一个标签和一个属性字典，它也有子节点，for child in root
        bboxes = []
        filename = root.find('filename').text #查找指定标签的文本内容，对于任何标签都可以有三个特征，标签名root.tag，标签属性root.attrib，标签的文本内容root.text
        for obj in root.iter('object'):       #迭代所有的object属性
            bbox = obj.find('bndbox')         #找到objecet中的 boundbox 坐标值                                                                         
            bbox = list(map(int, [bbox.find('xmin').text,
                                  bbox.find('ymin').text,
                                  bbox.find('xmax').text,
                                  bbox.find('ymax').text]))
            trkid = int(obj.find('trackid').text)
            if trkid in trajs:
                trajs[trkid].append(filename)#如果已经存在，就append
            else:#添加
                trajs[trkid] = [filename] 
            instance_img, _, _ = get_instance_image(img, bbox,
                    config.exemplar_size, config.instance_size, config.context_amount, img_mean)
            instance_img_name = os.path.join(save_folder, filename+".{:02d}.x.jpg".format(trkid))
            cv2.imwrite(instance_img_name, instance_img)
    return video_name, trajs

def processing(data_dir, output_dir, num_threads=32):
    # get all 4417 videos
    video_dir = os.path.join(data_dir, 'Data/VID') #./data/ILSVRC2015/Data/VID
    #获取训练集和测试集的路径 from glob import glob = glob.glob
    all_videos = glob(os.path.join(video_dir, 'train/ILSVRC2015_VID_train_0000/*')) + \
                 glob(os.path.join(video_dir, 'train/ILSVRC2015_VID_train_0001/*')) + \
                 glob(os.path.join(video_dir, 'train/ILSVRC2015_VID_train_0002/*')) + \
                 glob(os.path.join(video_dir, 'train/ILSVRC2015_VID_train_0003/*')) + \
                 glob(os.path.join(video_dir, 'val/*'))
    #all_videos = sorted(all_videos,key=lambda x: int(x.split('/')[-1].split('_')[-1]))
    meta_data = []
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #开启并行运算
        """ 
        funtional.partial
        可调用的partial对象，使用方法是partial(func,*args,**kw),func是必须要传入的

        imap和imap_unordered都用于对于大量数据遍历的多进程计算
        imap返回结果顺序和输入相同，imap_unordered则为不保证顺序 imap_unordered的返回迭代器的结果的排序是任意的,
        example:
            logging.info(pool.map(fun, range(10))) #把10个数放进函数里面，函数返回一个列表的结果
            pool.imap(fun,range(10))#把10个数放进fun里面，函数返回一个IMapIterator对象，每遍历一次，得到一个结果(进程由操作系统分配)
            pool.imap_unordered(fun,range(10)) #把10个数放进fun，函数返回一个IMapUnorderedIterator对象，每遍历一次得到一个结果
            
        Pool使用完毕后必须关闭，否则进程不会退出。有两种写法：
        (1)
        iter = pool.imap(func, iter)
        for ret in iter:
            #do something
        (2)推荐
        注意，第二种中，必须在with的块内使用iter
        with Pool() as pool:
            iter = pool.imap(func, iter)
            for ret in iter:
                #do something
        """
   
    with Pool(processes=num_threads) as pool:
        for ret in tqdm(pool.imap_unordered(functools.partial(worker, output_dir), all_videos), 
            total=len(all_videos)):
            meta_data.append(ret) #通过这种方式可以使得进程退出
    
     #开启单线 
    # 使用tqdm,一下三种方法都可以使用
    #for num in tqdm(range(0,len(all_videos))):
    #for video in tqdm(all_videos,total=len(all_videos)):
    # for video in tqdm(all_videos):
    #     ret=worker(output_dir, video)
    #     meta_data.append(ret)
    
    #save meta data
    pickle.dump(meta_data, open(os.path.join(output_dir, "meta_data.pkl"), 'wb'))

Data_dir='./data/ILSVRC2015'
Output_dir='./data/ILSVRC_VID_CURATION'
Num_threads=8# 原来设置为32

if __name__ == '__main__':
    # parse arguments
    parser=argparse.ArgumentParser(description="Demo SiamFC")
    parser.add_argument('--d',default=Data_dir,type=str,help="data_dir")
    parser.add_argument('--o',default=Output_dir,type=str, help="out put")
    parser.add_argument('--t',default=Num_threads, type=int, help="thread_num")
    args=parser.parse_args()
        
    processing(args.d, args.o, args.t)
    #Fire(processing) 
    '''
    原来都是使用argparse库进行命令行解析，需要在python文件的开头需要大量的代码设定各个命令行参数。
    而使用fire库不需要在python文件中设定命令行参数的代码，shell中指定函数名和对应参数即可
    train.py
        def train(a,b):
	    return a + b
    第一种调用方法
    train 1 2

    第二种调用方式
    train --a 1 --b 2
    
    '''

