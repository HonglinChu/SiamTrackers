
class Config:
    # dataset related
    exemplar_size = 127                    # exemplar size   z
    instance_size = 255                    # instance size   x
    context_amount= 0.5                    # context amount  比例

    # training related
    num_per_epoch = 53200                  # num of samples per epoch
    train_ratio   = 0.9                    # training ratio of VID dataset
    frame_range   = 100                    # frame range of choosing the instance
    train_batch_size = 8                   # training batch size
    valid_batch_size = 8                  # validation batch size
    train_num_workers =8                 # number of workers of train dataloader
    valid_num_workers =8                 # number of workers of validation dataloader
    lr = 1e-2                              # learning rate of SGD
    momentum = 0.0                         # momentum of SGD
    weight_decay = 0.0                     # weight decay of optimizator
    step_size = 25                         # step size of LR_Schedular
    gamma = 0.1                            # decay rate of LR_Schedular
    epoch = 30                             # total epoch
    seed = 1234                            # seed to sample training videos
    log_dir = './models/logs'              # log dirs
    radius = 16                            # radius of positive label
    response_scale = 1e-3                  # normalize of response
    max_translate = 3                      # max translation of random shift

    # tracking related
    scale_step = 1.0375                    # scale step of instance image
    num_scale = 3                          # number of scales
    scale_lr = 0.59                        # scale learning rate
    response_up_stride = 16                # response upsample stride
    response_sz = 17                       # response size
    train_response_sz = 15                 # train response size
    window_influence = 0.176               # window influence
    scale_penalty = 0.9745                 # scale penalty
    total_stride = 8                       # total stride of backbone
    sample_type = 'uniform'
    gray_ratio = 0.25
    blur_ratio = 0.15

    #test related
    model_path='./models/siamfc_30.pth'
    gpu_id= 1
    
config = Config()
