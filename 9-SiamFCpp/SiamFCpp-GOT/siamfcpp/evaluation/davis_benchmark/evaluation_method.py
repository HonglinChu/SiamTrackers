# -*- coding: utf-8 -*
import sys
from time import time

import numpy as np
import pandas as pd

from ..davis_benchmark.davis2017.davis_evaluation import DAVISEvaluation

dataset = 'val'
task = 'semi-supervised'


def davis2017_eval(davis_path,
                   results_path,
                   csv_name_global_path,
                   csv_name_per_sequence_path,
                   hp_dict,
                   version='2017'):
    time_start = time()
    hp_keys = list(hp_dict.keys())
    hp_values = [hp_dict[k] for k in hp_keys]

    print('Evaluating sequences for the {} task...'.format(task))
    # Create dataset and evaluate
    dataset_eval = DAVISEvaluation(davis_root=davis_path,
                                   task=task,
                                   gt_set=dataset,
                                   version=version)
    metrics_res = dataset_eval.evaluate(results_path)
    J, F = metrics_res['J'], metrics_res['F']

    # Generate dataframe for the general results
    g_measures = [
        'J&F-Mean', 'J-Mean', 'J-Recall', 'J-Decay', 'F-Mean', 'F-Recall',
        'F-Decay'
    ] + hp_keys
    final_mean = (np.mean(J["M"]) + np.mean(F["M"])) / 2.
    g_res = np.array([
        final_mean,
        np.mean(J["M"]),
        np.mean(J["R"]),
        np.mean(J["D"]),
        np.mean(F["M"]),
        np.mean(F["R"]),
        np.mean(F["D"])
    ] + hp_values)
    g_res = np.reshape(g_res, [1, len(g_res)])
    table_g = pd.DataFrame(data=g_res, columns=g_measures)
    with open(csv_name_global_path, 'a') as f:
        table_g.to_csv(f, index=False, float_format="%.3f", mode='a')
    print('Global results saved in {}'.format(csv_name_global_path))

    seq_names = list(J['M_per_object'].keys())
    seq_measures = ['Sequence', 'J-Mean', 'F-Mean']
    J_per_object = [J['M_per_object'][x] for x in seq_names]
    F_per_object = [F['M_per_object'][x] for x in seq_names]
    table_seq = pd.DataFrame(data=list(
        zip(seq_names, J_per_object, F_per_object)),
                             columns=seq_measures)
    with open(csv_name_per_sequence_path, 'w') as f:
        table_seq.to_csv(f, index=False, float_format="%.3f")
    print('Per-sequence results saved in {}'.format(csv_name_per_sequence_path))

    # Print the results
    sys.stdout.write(
        "--------------------------- Global results for {} ---------------------------\n"
        .format(dataset))
    print(table_g.to_string(index=False))
    sys.stdout.write(
        "\n---------- Per sequence results for {} ----------\n".format(dataset))
    print(table_seq.to_string(index=False))
    total_time = time() - time_start
    sys.stdout.write('\nTotal time:' + str(total_time))
    return dict(JF=final_mean,
                JM=np.mean(J["M"]),
                JR=np.mean(J["R"]),
                JD=np.mean(J["D"]),
                FM=np.mean(F["M"]),
                FR=np.mean(F["R"]),
                FD=np.mean(F["D"]))
