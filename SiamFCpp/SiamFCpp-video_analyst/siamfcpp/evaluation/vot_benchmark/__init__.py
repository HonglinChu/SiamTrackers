from .bbox_helper import cxy_wh_2_rect, get_axis_aligned_bbox
from .benchmark_helper import get_img, load_dataset
from .pysot.datasets import VOTDataset
# from .pysot.evaluation import AccuracyRobustnessBenchmark, EAOBenchmark
# from .pysot.utils.region import vot_float2str, vot_overlap


def show_result(self, result, topk=10, result_file=None):
    """pretty result_file.write result
    Args:
        result: returned dict from function eval
    """
    if len(self.tags) == 1:
        tracker_name_len = max((max([len(x) for x in result.keys()]) + 2), 12)
        header = ("|{:^" + str(tracker_name_len) + "}|{:^10}|").format(
            'Tracker Name', 'EAO')
        bar = '-' * len(header)
        formatter = "|{:^20}|{:^10.3f}|"
        result_file.write(bar + '\n')
        result_file.write(header + '\n')
        result_file.write(bar + '\n')
        tracker_eao = sorted(result.items(),
                             key=lambda x: x[1]['all'],
                             reverse=True)[:topk]
        for tracker_name, eao in tracker_eao:
            result_file.write(formatter.format(tracker_name, eao) + '\n')
        result_file.write(bar + '\n')
    else:
        header = "|{:^20}|".format('Tracker Name')
        header += "{:^7}|{:^15}|{:^14}|{:^15}|{:^13}|{:^11}|{:^7}|".format(
            *self.tags)
        bar = '-' * len(header)
        formatter = "{:^7.3f}|{:^15.3f}|{:^14.3f}|{:^15.3f}|{:^13.3f}|{:^11.3f}|{:^7.3f}|"
        result_file.write(bar + '\n')
        result_file.write(header + '\n')
        result_file.write(bar + '\n')
        sorted_tacker = sorted(result.items(),
                               key=lambda x: x[1]['all'],
                               reverse=True)[:topk]
        sorted_tacker = [x[0] for x in sorted_tacker]
        for tracker_name in sorted_tacker:
            result_file.write(
                "|{:^20}|".format(tracker_name) +
                formatter.format(*[result[tracker_name][x]
                                   for x in self.tags]) + '\n')
        result_file.write(bar + '\n')
