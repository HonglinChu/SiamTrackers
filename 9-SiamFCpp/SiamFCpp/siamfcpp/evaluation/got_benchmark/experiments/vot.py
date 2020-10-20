from __future__ import absolute_import, division, print_function

import time
import numpy as np
import os
import glob
import warnings
import json
from PIL import Image

from ..datasets import VOT
from ..utils.metrics import poly_iou
from ..utils.viz import show_frame


class ExperimentVOT(object):
    r"""Experiment pipeline and evaluation toolkit for VOT dataset.

    Notes:
        - The tracking results of three types of experiments ``supervised``
            ``unsupervised`` and ``realtime`` are compatible with the official
            VOT toolkit <https://github.com/votchallenge/vot-toolkit/>`.
        - TODO: The evaluation function for VOT tracking results is still
            under development.
    
    Args:
        root_dir (string): Root directory of VOT dataset where sequence
            folders exist.
        version (integer, optional): Specify the VOT dataset version. Specify as
            one of 2013~2018. Default is 2017.
        list_file (string, optional): If provided, only run experiments over
            sequences specified by the file.
        read_image (boolean, optional): If True, return the read PIL image in
            each frame. Otherwise only return the image path. Default is True.
        experiments (string or tuple): Specify the type(s) of experiments to run.
            Default is a tuple (``supervised``, ``unsupervised``, ``realtime``).
        result_dir (string, optional): Directory for storing tracking
            results. Default is ``./results``.
        report_dir (string, optional): Directory for storing performance
            evaluation results. Default is ``./reports``.
    """
    def __init__(self, root_dir, version=2017,
                 read_image=True, list_file=None,
                 experiments=('supervised', 'unsupervised', 'realtime'),
                 result_dir='results', report_dir='reports'):
        super(ExperimentVOT, self).__init__()
        if isinstance(experiments, str):
            experiments = (experiments,)
        assert all([e in ['supervised', 'unsupervised', 'realtime']
                    for e in experiments])
        self.dataset = VOT(
            root_dir, version, anno_type='default',
            download=True, return_meta=True, list_file=list_file)
        self.experiments = experiments
        if version == 'LT2018':
            version = '-' + version
        self.read_image = read_image
        self.result_dir = os.path.join(result_dir, 'VOT' + str(version))
        self.report_dir = os.path.join(report_dir, 'VOT' + str(version))
        self.skip_initialize = 5
        self.burnin = 10
        self.repetitions = 15
        self.sensitive = 100
        self.nbins_eao = 1500
        self.tags = ['camera_motion', 'illum_change', 'occlusion',
                     'size_change', 'motion_change', 'empty']

    def run(self, tracker, visualize=False):
        print('Running tracker %s on %s...' % (
            tracker.name, type(self.dataset).__name__))

        # run all specified experiments
        if 'supervised' in self.experiments:
            self.run_supervised(tracker, visualize)
        if 'unsupervised' in self.experiments:
            self.run_unsupervised(tracker, visualize)
        if 'realtime' in self.experiments:
            self.run_realtime(tracker, visualize)

    def run_supervised(self, tracker, visualize=False):
        print('Running supervised experiment...')

        # loop over the complete dataset
        for s, (img_files, anno, _) in enumerate(self.dataset):
            seq_name = self.dataset.seq_names[s]
            print('--Sequence %d/%d: %s' % (s + 1, len(self.dataset), seq_name))

            # rectangular bounding boxes
            anno_rects = anno.copy()
            if anno_rects.shape[1] == 8:
                anno_rects = self.dataset._corner2rect(anno_rects)

            # run multiple repetitions for each sequence
            for r in range(self.repetitions):
                # check if the tracker is deterministic
                if r > 0 and tracker.is_deterministic:
                    break
                elif r == 3 and self._check_deterministic('baseline', tracker.name, seq_name):
                    print('  Detected a deterministic tracker, ' +
                          'skipping remaining trials.')
                    break
                print(' Repetition: %d' % (r + 1))

                # skip if results exist
                record_file = os.path.join(
                    self.result_dir, tracker.name, 'baseline', seq_name,
                    '%s_%03d.txt' % (seq_name, r + 1))
                if os.path.exists(record_file):
                    print('  Found results, skipping', seq_name)
                    continue

                # state variables
                boxes = []
                times = []
                failure = False
                next_start = -1

                # tracking loop
                for f, img_file in enumerate(img_files):
                    image = Image.open(img_file)
                    if self.read_image:
                        frame = image
                    else:
                        frame = img_file

                    start_time = time.time()
                    if f == 0:
                        # initial frame
                        tracker.init(frame, anno_rects[0])
                        boxes.append([1])
                    elif failure:
                        # during failure frames
                        if f == next_start:
                            failure = False
                            tracker.init(frame, anno_rects[f])
                            boxes.append([1])
                        else:
                            start_time = np.NaN
                            boxes.append([0])
                    else:
                        # during success frames
                        box = tracker.update(frame)
                        iou = poly_iou(anno[f], box, bound=image.size)
                        if iou <= 0.0:
                            # tracking failure
                            failure = True
                            next_start = f + self.skip_initialize
                            boxes.append([2])
                        else:
                            # tracking succeed
                            boxes.append(box)
                    
                    # store elapsed time
                    times.append(time.time() - start_time)

                    # visualize if required
                    if visualize:
                        if len(boxes[-1]) == 4:
                            show_frame(image, boxes[-1])
                        else:
                            show_frame(image)
                
                # record results
                self._record(record_file, boxes, times)

    def run_unsupervised(self, tracker, visualize=False):
        print('Running unsupervised experiment...')

        # loop over the complete dataset
        for s, (img_files, anno, _) in enumerate(self.dataset):
            seq_name = self.dataset.seq_names[s]
            print('--Sequence %d/%d: %s' % (s + 1, len(self.dataset), seq_name))

            # skip if results exist
            record_file = os.path.join(
                self.result_dir, tracker.name, 'unsupervised', seq_name,
                '%s_001.txt' % seq_name)
            if os.path.exists(record_file):
                print('  Found results, skipping', seq_name)
                continue

            # rectangular bounding boxes
            anno_rects = anno.copy()
            if anno_rects.shape[1] == 8:
                anno_rects = self.dataset._corner2rect(anno_rects)

            # tracking loop
            boxes, times = tracker.track(
                img_files, anno_rects[0], visualize=visualize)
            assert len(boxes) == len(anno)

            # re-formatting
            boxes = list(boxes)
            boxes[0] = [1]
            
            # record results
            self._record(record_file, boxes, times)

    def run_realtime(self, tracker, visualize=False):
        print('Running real-time experiment...')

        # loop over the complete dataset
        for s, (img_files, anno, _) in enumerate(self.dataset):
            seq_name = self.dataset.seq_names[s]
            print('--Sequence %d/%d: %s' % (s + 1, len(self.dataset), seq_name))

            # skip if results exist
            record_file = os.path.join(
                self.result_dir, tracker.name, 'realtime', seq_name,
                '%s_001.txt' % seq_name)
            if os.path.exists(record_file):
                print('  Found results, skipping', seq_name)
                continue

            # rectangular bounding boxes
            anno_rects = anno.copy()
            if anno_rects.shape[1] == 8:
                anno_rects = self.dataset._corner2rect(anno_rects)

            # state variables
            boxes = []
            times = []
            next_start = 0
            failure = False
            failed_frame = -1
            total_time = 0.0
            grace = 3 - 1
            offset = 0

            # tracking loop
            for f, img_file in enumerate(img_files):
                image = Image.open(img_file)
                if self.read_image:
                    frame = image
                else:
                    frame = img_file

                start_time = time.time()
                if f == next_start:
                    # during initial frames
                    tracker.init(frame, anno_rects[f])
                    boxes.append([1])

                    # reset state variables
                    failure = False
                    failed_frame = -1
                    total_time = 0.0
                    grace = 3 - 1
                    offset = f
                elif not failure:
                    # during success frames
                    # calculate current frame
                    if grace > 0:
                        total_time += 1000.0 / 25
                        grace -= 1
                    else:
                        total_time += max(1000.0 / 25, last_time * 1000.0)
                    current = offset + int(np.round(np.floor(total_time * 25) / 1000.0))

                    # delayed/tracked bounding box
                    if f < current:
                        box = boxes[-1]
                    elif f == current:
                        box = tracker.update(frame)

                    iou = poly_iou(anno[f], box, bound=image.size)
                    if iou <= 0.0:
                        # tracking failure
                        failure = True
                        failed_frame = f
                        next_start = current + self.skip_initialize
                        boxes.append([2])
                    else:
                        # tracking succeed
                        boxes.append(box)
                else:
                    # during failure frames
                    if f < current:
                        # skipping frame due to slow speed
                        boxes.append([0])
                        start_time = np.NaN
                    elif f == current:
                        # current frame
                        box = tracker.update(frame)
                        iou = poly_iou(anno[f], box, bound=image.size)
                        if iou <= 0.0:
                            # tracking failure
                            boxes.append([2])
                            boxes[failed_frame] = [0]
                            times[failed_frame] = np.NaN
                        else:
                            # tracking succeed
                            boxes.append(box)
                    elif f < next_start:
                        # skipping frame due to failure
                        boxes.append([0])
                        start_time = np.NaN

                # store elapsed time
                last_time = time.time() - start_time
                times.append(last_time)

                # visualize if required
                if visualize:
                    if len(boxes[-1]) == 4:
                        show_frame(image, boxes[-1])
                    else:
                        show_frame(image)

            # record results
            self._record(record_file, boxes, times)

    def report(self, tracker_names):
        assert isinstance(tracker_names, (list, tuple))

        # function for loading results
        def read_record(filename):
            with open(filename) as f:
                record = f.read().strip().split('\n')
            record = [[float(t) for t in line.split(',')]
                      for line in record]
            return record

        # assume tracker_names[0] is your tracker
        report_dir = os.path.join(self.report_dir, tracker_names[0])
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)
        report_file = os.path.join(report_dir, 'performance.json')

        performance = {}
        for name in tracker_names:
            print('Evaluating', name)
            ious = {}
            ious_full = {}
            failures = {}
            times = {}
            masks = {}  # frame masks for attribute tags

            for s, (img_files, anno, meta) in enumerate(self.dataset):
                seq_name = self.dataset.seq_names[s]

                # initialize frames scores
                frame_num = len(img_files)
                ious[seq_name] = np.full(
                    (self.repetitions, frame_num), np.nan, dtype=float)
                ious_full[seq_name] = np.full(
                    (self.repetitions, frame_num), np.nan, dtype=float)
                failures[seq_name] = np.full(
                    (self.repetitions, frame_num), np.nan, dtype=float)
                times[seq_name] = np.full(
                    (self.repetitions, frame_num), np.nan, dtype=float)

                # read results of all repetitions
                record_files = sorted(glob.glob(os.path.join(
                    self.result_dir, name, 'baseline', seq_name,
                    '%s_[0-9]*.txt' % seq_name)))
                boxes = [read_record(f) for f in record_files]
                assert all([len(b) == len(anno) for b in boxes])

                # calculate frame ious with burnin
                bound = Image.open(img_files[0]).size
                seq_ious = [self._calc_iou(b, anno, bound, burnin=True)
                            for b in boxes]
                ious[seq_name][:len(seq_ious), :] = seq_ious

                # calculate frame ious without burnin
                seq_ious_full = [self._calc_iou(b, anno, bound)
                                 for b in boxes]
                ious_full[seq_name][:len(seq_ious_full), :] = seq_ious_full

                # calculate frame failures
                seq_failures = [
                    [len(b) == 1 and b[0] == 2 for b in boxes_per_rep]
                    for boxes_per_rep in boxes]
                failures[seq_name][:len(seq_failures), :] = seq_failures

                # collect frame runtimes
                time_file = os.path.join(
                    self.result_dir, name, 'baseline', seq_name,
                    '%s_time.txt' % seq_name)
                if os.path.exists(time_file):
                    seq_times = np.loadtxt(time_file, delimiter=',').T
                    times[seq_name][:len(seq_times), :] = seq_times

                # collect attribute masks
                tag_num = len(self.tags)
                masks[seq_name] = np.zeros((tag_num, frame_num), bool)
                for i, tag in enumerate(self.tags):
                    if tag in meta:
                        masks[seq_name][i, :] = meta[tag]
                # frames with no tags
                if 'empty' in self.tags:
                    tag_frames = np.array([
                        v for k, v in meta.items()
                        if not 'practical' in k], dtype=bool)
                    ind = self.tags.index('empty')
                    masks[seq_name][ind, :] = \
                        ~np.logical_or.reduce(tag_frames, axis=0)

            # concatenate frames
            seq_names = self.dataset.seq_names
            masks = np.concatenate(
                [masks[s] for s in seq_names], axis=1)
            ious = np.concatenate(
                [ious[s] for s in seq_names], axis=1)
            failures = np.concatenate(
                [failures[s] for s in seq_names], axis=1)

            with warnings.catch_warnings():
                # average over repetitions
                warnings.simplefilter('ignore', category=RuntimeWarning)
                ious = np.nanmean(ious, axis=0)
                failures = np.nanmean(failures, axis=0)
            
                # calculate average overlaps and failures for each tag
                tag_ious = np.array(
                    [np.nanmean(ious[m]) for m in masks])
                tag_failures = np.array(
                    [np.nansum(failures[m]) for m in masks])
                tag_frames = masks.sum(axis=1)

            # remove nan values
            tag_ious[np.isnan(tag_ious)] = 0.0
            tag_weights = tag_frames / tag_frames.sum()

            # calculate weighted accuracy and robustness
            accuracy = np.sum(tag_ious * tag_weights)
            robustness = np.sum(tag_failures * tag_weights)

            # calculate tracking speed
            times = np.concatenate([
                t.reshape(-1) for t in times.values()])
            # remove invalid values
            times = times[~np.isnan(times)]
            times = times[times > 0]
            if len(times) > 0:
                speed = np.mean(1. / times)
            else:
                speed = -1

            performance.update({name: {
                'accuracy': accuracy,
                'robustness': robustness,
                'speed_fps': speed}})

        # save performance
        with open(report_file, 'w') as f:
            json.dump(performance, f, indent=4)
        print('Performance saved at', report_file)

        return performance

    def show(self, tracker_names, seq_names=None, play_speed=1,
             experiment='supervised'):
        if seq_names is None:
            seq_names = self.dataset.seq_names
        elif isinstance(seq_names, str):
            seq_names = [seq_names]
        assert isinstance(tracker_names, (list, tuple))
        assert isinstance(seq_names, (list, tuple))
        assert experiment in ['supervised', 'unsupervised', 'realtime']

        play_speed = int(round(play_speed))
        assert play_speed > 0

        # "supervised" experiment results are stored in "baseline" folder
        if experiment == 'supervised':
            experiment = 'baseline'

        # function for loading results
        def read_record(filename):
            with open(filename) as f:
                record = f.read().strip().split('\n')
            record = [[float(t) for t in line.split(',')]
                      for line in record]
            for i, r in enumerate(record):
                if len(r) == 4:
                    record[i] = np.array(r)
                elif len(r) == 8:
                    r = np.array(r)[np.newaxis, :]
                    r = self.dataset._corner2rect(r)
                    record[i] = r[0]
                else:
                    record[i] = np.zeros(4)
            return record

        for s, seq_name in enumerate(seq_names):
            print('[%d/%d] Showing results on %s...' % (
                s + 1, len(seq_names), seq_name))
            
            # load all tracking results
            records = {}
            for name in tracker_names:
                record_file = os.path.join(
                    self.result_dir, name, experiment, seq_name,
                    '%s_001.txt' % seq_name)
                records[name] = read_record(record_file)

            # loop over the sequence and display results
            img_files, anno, _ = self.dataset[seq_name]
            if anno.shape[1] == 8:
                anno = self.dataset._corner2rect(anno)
            for f, img_file in enumerate(img_files):
                if not f % play_speed == 0:
                    continue
                image = Image.open(img_file)
                boxes = [anno[f]] + [
                    records[name][f] for name in tracker_names]
                show_frame(image, boxes,
                           legends=['GroundTruth'] + tracker_names,
                           colors=['w', 'r', 'g', 'b', 'c', 'm', 'y',
                                   'orange', 'purple', 'brown', 'pink'])

    def _record(self, record_file, boxes, times):
        # convert boxes to string
        lines = []
        for box in boxes:
            if len(box) == 1:
                lines.append('%d' % box[0])
            else:
                lines.append(str.join(',', ['%.4f' % t for t in box]))

        # record bounding boxes
        record_dir = os.path.dirname(record_file)
        if not os.path.isdir(record_dir):
            os.makedirs(record_dir)
        with open(record_file, 'w') as f:
            f.write(str.join('\n', lines))
        print('  Results recorded at', record_file)

        # convert times to string
        lines = ['%.4f' % t for t in times]
        lines = [t.replace('nan', 'NaN') for t in lines]

        # record running times
        time_file = record_file[:record_file.rfind('_')] + '_time.txt'
        if os.path.exists(time_file):
            with open(time_file) as f:
                exist_lines = f.read().strip().split('\n')
            lines = [t + ',' + s for t, s in zip(exist_lines, lines)]
        with open(time_file, 'w') as f:
            f.write(str.join('\n', lines))

    def _check_deterministic(self, exp, tracker_name, seq_name):
        record_dir = os.path.join(
            self.result_dir, tracker_name, exp, seq_name)
        record_files = sorted(glob.glob(os.path.join(
            record_dir, '%s_[0-9]*.txt' % seq_name)))

        if len(record_files) < 3:
            return False
        
        records = []
        for record_file in record_files:
            with open(record_file, 'r') as f:
                records.append(f.read())
        
        return len(set(records)) == 1

    def _calc_iou(self, boxes, anno, bound, burnin=False):
        # skip initialization frames
        if burnin:
            boxes = boxes.copy()
            init_inds = [i for i, box in enumerate(boxes)
                         if box == [1.0]]
            for ind in init_inds:
                boxes[ind:ind + self.burnin] = [[0]] * self.burnin
        # calculate polygon ious
        ious = np.array([poly_iou(np.array(a), b, bound)
                         if len(a) > 1 else np.NaN
                         for a, b in zip(boxes, anno)])
        return ious
