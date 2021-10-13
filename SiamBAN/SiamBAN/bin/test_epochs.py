import sys
sys.path.append("..")
import os
import time
import argparse
from mpi4py import MPI


parser = argparse.ArgumentParser(description="multi-gpu test all epochs")
parser.add_argument("--start_epoch", default=1, type=int, required=True, help="test end epoch")
parser.add_argument("--end_epoch", default=20, type=int, required=True, help="test end epoch")
parser.add_argument("--gpu_nums", default=3, type=int, required=True, help="gpu numbers")
parser.add_argument("--threads", default=3, type=int, required=True)
parser.add_argument("--dataset", default="OTB2015", type=str, help="benchmark to test")
args = parser.parse_args()

# init gpu and epochs
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
GPU_ID = rank % args.gpu_nums
node_name = MPI.Get_processor_name()  # get the name of the node
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
print("node name: {}, GPU_ID: {}".format(node_name, GPU_ID))
time.sleep(rank * 5)

# run test scripts -- one epoch for each thread
for i in range((args.end_epoch - args.start_epoch + 1) // args.threads + 1):
    dataset = args.dataset
    try:
        epoch_ID += args.threads
    except:
        epoch_ID = rank % (args.end_epoch - args.start_epoch + 1) + args.start_epoch

    if epoch_ID > args.end_epoch:
        continue

    snapshot = "snapshot/checkpoint_e{}.pth".format(epoch_ID)
    print("==> test {}th epoch".format(epoch_ID))

    os.system("python ../../tools/test.py --snapshot {0} --dataset {1} --config config.yaml".format(snapshot, dataset))
