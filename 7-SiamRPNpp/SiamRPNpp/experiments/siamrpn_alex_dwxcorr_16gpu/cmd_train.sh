CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=2333 \
    ../../bin/train.py  --cfg config.yaml