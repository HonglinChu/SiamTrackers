import torch
def compute_locations(features,stride): # [B, 2, 25, 25],8 
    h, w = features.size()[-2:]
    locations_per_level = compute_locations_per_level(
        h, w, stride,
        features.device
    )
    return locations_per_level 

def compute_locations_per_level(h, w, stride, device):# [25, 25, 8, cuda]
    # [0, 8, 16, 24, ..., 192] # 25 resnet 
    shifts_x = torch.arange(
        0, w * stride, step=stride,
        dtype=torch.float32, device=device
    ) 
    
    # [0, 8, 16, 24, ..., 192] # 25 个 resnet
    shifts_y = torch.arange(
        0, h * stride, step=stride,
        dtype=torch.float32, device=device
    )
    
    shift_y, shift_x = torch.meshgrid((shifts_y, shifts_x))#[0,8,16] [0,8,16]-->[[0,0,0],[8,8,8],[16,16,16]] , [[0,8,16],[0,8,16],[0,8,16]]
    
    shift_x = shift_x.reshape(-1) # [25, 25] --> 625  [0,8,16,0,8,16,0,8,16]
    
    shift_y = shift_y.reshape(-1) # [25, 25] --> 625  [0,0,0,8,8,8,16,16,16]
    # torch.cat() 对tensor沿指定维度拼接，但返回的Tensor的维数不变；torch.stack() 对tensor沿着指定维度拼接，但是返回的维度会多一维。
    #tmp=torch.stack((shift_x, shift_y), dim=1) # [[0,0],[8,0],[16,0],[0,8],[8,8],[16,8],[0,16],[8,16],[16,16]]

    # resnet50
    #locations = torch.stack((shift_x, shift_y), dim=1) + 32  #resnet:32  // 32  [[32,32],[40,32],[48,32],[32,40],[40,40],[48,40],[32,48],[40,48],[48,48]]
   
    # alexnet 
    locations = torch.stack((shift_x, shift_y), dim=1) + 63  #alex:63  

    return locations
