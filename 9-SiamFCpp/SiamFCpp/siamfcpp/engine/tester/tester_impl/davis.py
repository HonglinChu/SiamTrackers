# -*- coding: utf-8 -*
import copy
import itertools
import math
import os
from os import makedirs
from os.path import isdir, join

import cv2
import numpy as np
from loguru import logger
from PIL import Image
from tqdm import tqdm

import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Manager, Pool

from siamfcpp.evaluation import davis_benchmark
from siamfcpp.utils import ensure_dir

from ..tester_base import TRACK_TESTERS, VOS_TESTERS, TesterBase


@VOS_TESTERS.register
class DAVISTester(TesterBase):
    r"""
    Tester to test the davis2017 dataset, the result is saved as follows
    exp_dir/logs/$dataset_name$/$tracker_name/
                                    |-baseline/$video_name$/ folder of result files
                                    |-eval_result.csv evaluation result file

    Hyper-parameters
    ----------------
    device_num: int
        number of gpu for test
    data_root: str
        davis2017 dataset root directory. dict(dataset_name: path_to_root)
    dataset_names: str
        daataset name (DAVIS2017)
    save_video: bool
        save videos with predicted mask overlap for visualization and debug
    save_patch: bool
    """

    extra_hyper_params = dict(device_num=1,
                              data_root="datasets/DAVIS",
                              dataset_names=[
                                  "DAVIS2017",
                              ],
                              save_video=False,
                              save_patch=False)

    def __init__(self, *args, **kwargs):
        r"""
        Crete tester with config and pipeline

        Arguments
        ---------
        cfg: CfgNode
            parent config, (e.g. model / pipeline / tester)
        pipeline: PipelineBase
            pipeline to test
        """
        super(DAVISTester, self).__init__(*args, **kwargs)
        self._state['speed'] = -1
        self.iou_eval_thres = np.arange(0.3, 0.5, 0.05)

    def test(self):
        r"""
        Run test
        """
        # set dir
        self.tracker_name = self._hyper_params["exp_name"]
        for dataset_name in self._hyper_params["dataset_names"]:
            self.dataset_name = dataset_name
            self.tracker_dir = os.path.join(self._hyper_params["exp_save"],
                                            self.dataset_name)
            self.save_root_dir = os.path.join(self.tracker_dir,
                                              self.tracker_name, "baseline")
            ensure_dir(self.save_root_dir)
            # track videos
            self.run_tracker()
            # evaluation
            eval_result = self.evaluation('default_hp')
        return dict(main_performance=eval_result["JF"])

    def run_tracker(self):
        """
        Run self.pipeline on DAVIS
        """
        num_gpu = self._hyper_params["device_num"]
        all_devs = [torch.device("cuda:%d" % i) for i in range(num_gpu)]
        logger.info('runing test on devices {}'.format(all_devs))
        davis_root = self._hyper_params["data_root"]
        logger.info('Using dataset %s at: %s' % (self.dataset_name, davis_root))
        # setup dataset
        dataset = davis_benchmark.load_dataset(davis_root, self.dataset_name)
        self.dataset = dataset
        keys = list(dataset.keys())
        keys.sort()
        nr_records = len(keys)
        pbar = tqdm(total=nr_records)
        mean_speed = -1
        speed_list = []
        manager = Manager()
        speed_queue = manager.Queue(500)
        # set worker
        if num_gpu == 0:
            self.worker(keys, all_devs[0], self.dataset, speed_queue)
            for i in range(nr_records):
                s = speed_queue.get()
                speed_list.append(s)
                pbar.update(1)
        else:
            nr_video = math.ceil(nr_records / num_gpu)
            procs = []
            for i in range(num_gpu):
                start = i * nr_video
                end = min(start + nr_video, nr_records)
                split_records = keys[start:end]
                proc = mp.Process(target=self.worker,
                                  args=(split_records, all_devs[i],
                                        self.dataset, speed_queue))
                logger.info('process:%d, start:%d, end:%d' % (i, start, end))
                proc.start()
                procs.append(proc)
            for i in range(nr_records):
                s = speed_queue.get()
                speed_list.append(s)
                pbar.update(1)
            for p in procs:
                p.join()
        # print result
        mean_speed = float(np.mean(speed_list))
        logger.info('Mean Speed: {:.2f} FPS'.format(mean_speed))
        self._state['speed'] = mean_speed

    def worker(self, records, dev, dataset, speed_queue=None):
        tracker = self._pipeline
        tracker.set_device(dev)
        for v_id, video in enumerate(records):
            speed = self.track_single_video_vos(tracker, dataset[video])
            if speed_queue is not None:
                speed_queue.put_nowait(speed)

    def evaluation(self, search_task_name):
        r"""
        Run evaluation & write result to csv file under self.tracker_dir
        """

        results_path = join(self.save_root_dir, 'results_multi')
        davis_data_path = self._hyper_params["data_root"]

        eval_dump_path = join(self.save_root_dir, 'dump')
        if not isdir(eval_dump_path): makedirs(eval_dump_path)

        csv_name_global_path = join(eval_dump_path,
                                    search_task_name + '_global_results.csv')
        csv_name_per_sequence_path = join(
            eval_dump_path, search_task_name + '_name_per_sequence_results.csv')

        version = self.dataset_name[-4:]
        hp_dict = {}
        return davis_benchmark.davis2017_eval(davis_data_path,
                                              results_path,
                                              csv_name_global_path,
                                              csv_name_per_sequence_path,
                                              hp_dict,
                                              version=version)

    def track_single_video_vos(self, tracker, video, mot_enable=True):
        '''
        perfrom semi-supervised video object segmentation for single video
        :param tracker: tracker pipeline
        :param video: video info
        :param mot_enable:  if true, perform instance level segmentation on davis, otherwise semantic
        '''
        image_files = video['image_files']

        annos = [np.array(Image.open(x)) for x in video['anno_files']]
        if 'anno_init_files' in video:
            annos_init = [
                np.array(Image.open(x)) for x in video['anno_init_files']
            ]
        else:
            annos_init = [annos[0]]

        if not mot_enable:
            annos = [(anno > 0).astype(np.uint8) for anno in annos]
            annos_init = [(anno_init > 0).astype(np.uint8)
                          for anno_init in annos_init]

        if 'start_frame' in video:
            object_ids = [int(id) for id in video['start_frame']]
        else:
            object_ids = [o_id for o_id in np.unique(annos[0]) if o_id != 0]
            if len(object_ids) != len(annos_init):
                annos_init = annos_init * len(object_ids)
        object_num = len(object_ids)
        toc = 0
        pred_masks = np.zeros((object_num, len(image_files), annos[0].shape[0],
                               annos[0].shape[1])) - 1

        if self._hyper_params['save_video']:
            track_boxes = np.zeros((object_num, len(image_files), 4))
            track_mask_boxes = np.zeros((object_num, len(image_files), 4))
            track_mask_score = np.zeros((object_num, len(image_files)))
            track_score = np.zeros((object_num, len(image_files)))
            state_score = np.zeros((object_num, len(image_files)))
        if self._hyper_params['save_patch']:
            patch_list = []

        for obj_id, o_id in enumerate(object_ids):
            obj_patch_list = []
            logger.info('{} th object in video {}'.format(o_id, video['name']))
            if 'start_frame' in video:
                start_frame = video['start_frame'][str(o_id)]
                end_frame = video['end_frame'][str(o_id)]
            else:
                start_frame, end_frame = 0, len(image_files)

            for f, image_file in enumerate(tqdm(image_files)):
                im = cv2.imread(image_file)
                img_h, img_w = im.shape[0], im.shape[1]

                tic = cv2.getTickCount()
                if f == start_frame:  # init
                    mask = (annos_init[obj_id] == o_id).astype(np.uint8)
                    x, y, w, h = cv2.boundingRect((mask).astype(np.uint8))
                    tracker.init(im, np.array([x, y, w, h]), mask)
                elif end_frame >= f > start_frame:  # tracking
                    mask = tracker.update(im)
                    if self._hyper_params['save_video']:
                        rect_mask = tracker._state['mask_rect']
                        mask_score = tracker._state['conf_score']
                        track_boxes[obj_id, f, :] = tracker._state['track_box']
                        track_mask_boxes[obj_id, f, :] = rect_mask
                        track_mask_score[obj_id, f] = mask_score
                        track_score[obj_id, f] = tracker._state["track_score"]
                        state_score[obj_id, f] = tracker._state["state_score"]

                    if self._hyper_params['save_patch']:
                        patch = tracker._state['patch_prediction']
                        obj_patch_list.append(patch)

                toc += cv2.getTickCount() - tic
                if end_frame >= f >= start_frame:
                    pred_masks[obj_id, f, :, :] = mask
            if self._hyper_params['save_patch']:
                patch_list.append(obj_patch_list)
        toc /= cv2.getTickFrequency()

        if len(annos) == len(image_files):
            multi_mean_iou = davis_benchmark.MultiBatchIouMeter(
                self.iou_eval_thres,
                pred_masks,
                annos,
                start=video['start_frame'] if 'start_frame' in video else None,
                end=video['end_frame'] if 'end_frame' in video else None)

        for i in range(object_num):
            for j, thr in enumerate(self.iou_eval_thres):
                logger.info(
                    'Fusion Multi Object{:20s} IOU at {:.2f}: {:.4f}'.format(
                        video['name'] + '_' + str(i + 1), thr,
                        multi_mean_iou[i, j]))

        if self._hyper_params['save_patch']:
            video_path = join(self.save_root_dir, 'patches', video['name'])
            logger.info('save patches path: {}'.format(video_path))
            if not isdir(video_path): makedirs(video_path)
            for i in range(len(patch_list)):
                patch_images = patch_list[i]
                for frame_id, patch_image in enumerate(patch_images):
                    cv2.imwrite(
                        join(video_path, 'obj_{}_{}.png'.format(i, frame_id)),
                        patch_image)

        video_path = join(self.save_root_dir, 'results_multi', video['name'])
        logger.info('save mask path:{}'.format(video_path))
        if not isdir(video_path): makedirs(video_path)
        pred_mask_final = np.array(pred_masks)
        pred_mask_final = (
            np.argmax(pred_mask_final, axis=0).astype('uint8') +
            1) * (np.max(pred_mask_final, axis=0) >
                  tracker._hyper_params['mask_pred_thresh']).astype('uint8')
        for i in range(pred_mask_final.shape[0]):
            mask_label = pred_mask_final[i].astype(np.uint8)
            cv2.imwrite(
                join(video_path,
                     image_files[i].split('/')[-1].split('.')[0] + '.png'),
                mask_label)
        logger.info(
            '({:d}) Video: {:12s} Time: {:02.1f}s Speed: {:3.1f}fps'.format(
                o_id, video['name'], toc,
                f * len(object_ids) / toc))
        speed = f * len(object_ids) / toc
        logger.info("{} speed: {}".format(video['name'], speed))

        if self._hyper_params['save_video']:
            video_path = join(self.save_root_dir, 'save_video')
            if not isdir(video_path): makedirs(video_path)
            logger.info('save video as : {}'.format(video_path))

            VideoOut = cv2.VideoWriter(
                video_path + '/' + video['name'] + '.avi',
                cv2.VideoWriter_fourcc(*'MJPG'), 10.0, (img_w, img_h))
            for f, image_file in enumerate(image_files):
                img = cv2.imread(image_file)
                mask_f = pred_mask_final[f, :, :]
                img = davis_benchmark.overlay_semantic_mask(img,
                                                            mask_f,
                                                            alpha=0.3,
                                                            contour_thickness=1)
                for i in range(object_num):
                    rect = track_boxes[i, f]
                    rect = [int(l) for l in rect]

                    rect_mask = track_mask_boxes[i, f]
                    rect_mask = [int(l) for l in rect_mask]

                    mask_score = round(track_mask_score[i, f], 2)
                    track_score_ = round(track_score[i, f], 2)
                    state_score_ = round(state_score[i, f], 2)
                    color = davis_benchmark.labelcolormap(object_num + 1)[i + 1]
                    color_tuple = (int(color[0]), int(color[1]), int(color[2]))
                    cv2.putText(img,
                                'Frame : ' + str(f), (10, 20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (255, 255, 255),
                                thickness=2)

                    cv2.rectangle(img, (rect[0], rect[1]),
                                  (rect[0] + rect[2], rect[1] + rect[3]),
                                  color=color_tuple,
                                  thickness=2)

                    if rect_mask[0] > 0:
                        cv2.rectangle(img, (rect_mask[0], rect_mask[1]),
                                      (rect_mask[0] + rect_mask[2],
                                       rect_mask[1] + rect_mask[3]),
                                      color=(255, 255, 255),
                                      thickness=2)
                    if f > 0:
                        cv2.putText(img,
                                    'M {} T{} S {}'.format(
                                        mask_score, track_score_, state_score_),
                                    (rect[0], max(rect[1], 5) + 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    color=(0, 0, 255),
                                    thickness=2)

                VideoOut.write(img)
            VideoOut.release()
        return speed


DAVISTester.default_hyper_params = copy.deepcopy(
    DAVISTester.default_hyper_params)
DAVISTester.default_hyper_params.update(DAVISTester.extra_hyper_params)
