from IPython import embed

import numpy as np
import pickle
import os
import cv2
import functools
import xml.etree.ElementTree as ET
import sys
import multiprocessing as mp

from multiprocessing import Pool
from fire import Fire
from tqdm import tqdm
from glob import glob

sys.path.append(os.getcwd())
from siamrpn.config import config
from siamrpn.utils import get_instance_image, add_box_img

def worker(output_dir, video_dir):
    instance_crop_size = 500
    if 'YT-BB' in video_dir:
        image_names = glob(os.path.join(video_dir, '*.jpg'))
        image_names = sorted(image_names, key=lambda x: int(x.split('/')[-1].split('_')[1]))
        video_name = '_'.join(os.path.basename(video_dir).split('_')[:-1])

        with open('/dataset_ssd/std_xml_ytb/' + video_name + '.pkl', 'rb') as f:
            std_xml_dict = pickle.load(f)

        save_folder = os.path.join(output_dir, video_name)
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)

        trajs = {}
        for image_name in image_names:
            img = cv2.imread(image_name)
            h_img, w_img, _ = img.shape
            img_mean = tuple(map(int, img.mean(axis=(0, 1))))
            frame = image_name.split('_')[-2]

            if int(frame) == 0:
                anno = std_xml_dict[str(int(frame))]
            else:
                anno = std_xml_dict[frame]

            filename = '_'.join(image_name.split('/')[-1].split('_')[:-1])
            for class_id in anno.keys():
                for track_id in anno[class_id].keys():
                    class_name, present, xmin_scale, xmax_scale, ymin_scale, ymax_scale = anno[class_id][track_id]
                    new_track_id = class_id.zfill(3) + track_id.zfill(3)
                    bbox = np.array(list(map(float, [xmin_scale, xmax_scale, ymin_scale, ymax_scale]))) * np.array(
                        [w_img, w_img, h_img, h_img])
                    if present == 'present':
                        if new_track_id in trajs.keys():
                            trajs[new_track_id].append(filename)
                        else:
                            trajs[new_track_id] = [filename]
                        bbox = np.array(
                            [(bbox[1] + bbox[0]) / 2, (bbox[3] + bbox[2]) / 2, bbox[1] - bbox[0] + 1,
                             bbox[3] - bbox[2] + 1])
                        instance_img, w, h, _ = get_instance_image(img, bbox,
                                                                   config.exemplar_size, instance_crop_size,
                                                                   config.context_amount,
                                                                   img_mean)
                        instance_img_name = os.path.join(save_folder,
                                                         filename + ".{}.x_{:.2f}_{:.2f}_{:.0f}_{:.0f}.jpg".format(
                                                             new_track_id,
                                                             w, h, w_img, h_img))
                        cv2.imwrite(instance_img_name, instance_img)

                    elif present == 'absent':
                        continue

    else:
        image_names = glob(os.path.join(video_dir, '*.JPEG'))
        image_names = sorted(image_names, key=lambda x: int(x.split('/')[-1].split('.')[0]))
        video_name = video_dir.split('/')[-1]
        save_folder = os.path.join(output_dir, video_name)
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        trajs = {}
        for image_name in image_names:
            img = cv2.imread(image_name)
            h_img, w_img, _ = img.shape
            img_mean = tuple(map(int, img.mean(axis=(0, 1))))
            anno_name = image_name.replace('Data', 'Annotations')
            anno_name = anno_name.replace('JPEG', 'xml')
            tree = ET.parse(anno_name)
            root = tree.getroot()
            bboxes = []
            filename = root.find('filename').text
            for obj in root.iter('object'):
                bbox = obj.find('bndbox')
                bbox = list(map(int, [bbox.find('xmin').text,
                                      bbox.find('ymin').text,
                                      bbox.find('xmax').text,
                                      bbox.find('ymax').text]))
                trkid = int(obj.find('trackid').text)
                if trkid in trajs:
                    trajs[trkid].append(filename)
                else:
                    trajs[trkid] = [filename]
                bbox = np.array(
                    [(bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2, bbox[2] - bbox[0] + 1,
                     bbox[3] - bbox[1] + 1])

                instance_img, w, h, _ = get_instance_image(img, bbox,
                                                           config.exemplar_size, instance_crop_size,
                                                           config.context_amount,
                                                           img_mean)
                instance_img_name = os.path.join(save_folder,
                                                 filename + ".{:02d}.x_{:.2f}_{:.2f}_{:.0f}_{:.0f}.jpg".format(trkid, w,
                                                                                                               h, w_img,
                                                                                                               h_img))
                cv2.imwrite(instance_img_name, instance_img)
    return video_name, trajs


def processing(vid_dir, ytb_dir, output_dir, num_threads=mp.cpu_count()):
    # get all 4417 videos in vid and all video in ytbb
    vid_video_dir = os.path.join(vid_dir, 'data/VID')

    ytb_video_dir = ytb_dir 

    # -------------------------------------------------------------------------------------
    # all_videos = glob(os.path.join(ytb_video_dir, 'v*/youtube_dection_frame_temp/*'))
    # all_videos = glob('/mnt/diska1/YT-BB/v1/youtube_dection_frame_temp/130dH0FNXio_*')
    # -------------------------------------------------------------------------------------

    all_videos = glob(os.path.join(vid_video_dir, 'train/ILSVRC2015_VID_train_0000/*')) + \
                 glob(os.path.join(vid_video_dir, 'train/ILSVRC2015_VID_train_0001/*')) + \
                 glob(os.path.join(vid_video_dir, 'train/ILSVRC2015_VID_train_0002/*')) + \
                 glob(os.path.join(vid_video_dir, 'train/ILSVRC2015_VID_train_0003/*')) + \
                 glob(os.path.join(vid_video_dir, 'val/*'))+\
                 glob(os.path.join(ytb_video_dir, 'v*/youtube_dection_frame_temp/*'))

    meta_data = []
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    all_videos = [x for x in all_videos if 'imagelist' not in x]

    # for video in tqdm(all_videos):
    # functools.partial(worker, output_dir)(video)

    # -------------------------------------------------------------------------------------
    # load former meta_data
    # with open('/dataset_ssd/ytb_vid_rpn/meta_data.pkl', 'rb') as f:
    #     former_pkl = pickle.load(f)
    # former_dict = {x[0]: x[1] for x in former_pkl if 'ILSVRC2015' in x[0]}
    # former_vid = []
    # for k in former_dict.keys():
    #     former_vid.append((k, former_dict[k]))
    # meta_data.extend(former_vid)
    # -------------------------------------------------------------------------------------

    with Pool(processes=num_threads) as pool:
        for ret in tqdm(pool.imap_unordered(
                functools.partial(worker, output_dir), all_videos), total=len(all_videos)):
            meta_data.append(ret)
            tqdm.write(ret[0])

    # save meta data
    pickle.dump(meta_data, open(os.path.join(output_dir, "meta_data.pkl"), 'wb'))


if __name__ == '__main__':
    Fire(processing)
