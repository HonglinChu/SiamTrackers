import numpy as np
from IPython import embed
#获取所有像素点的anchors
#base_size=8; scales=8; ratios=0.33, 0.5, 1, 2, 3
def generate_anchors(total_stride, base_size, scales, ratios, score_size):
    anchor_num = len(ratios) * len(scales)
    anchor = np.zeros((anchor_num, 4), dtype=np.float32)
    size = base_size * base_size
    count = 0
    for ratio in ratios:
        # ws = int(np.sqrt(size * 1.0 / ratio))
        ws = int(np.sqrt(size / ratio))
        hs = int(ws * ratio)
        for scale in scales:
            wws = ws * scale
            hhs = hs * scale
            anchor[count, 0] = 0
            anchor[count, 1] = 0
            anchor[count, 2] = wws
            anchor[count, 3] = hhs
            count += 1
    anchor= np.tile(anchor, score_size * score_size)# [5,1444]
    #tile是瓷砖的意思，就是将原矩阵横向、纵向地复制 tile（matrix，（1,4)=tile（matrix，4）横向复制
    anchor = anchor.reshape((-1, 4))#[5x17x17,4]  #
    # (5,17x17x4) to (17x17x5,4)
    ori = - (score_size // 2) * total_stride #??
    # the left displacement
    xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                         [ori + total_stride * dy for dy in range(score_size)])
    # (15,15) or (17,17) or (19,19)
    xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), np.tile(yy.flatten(), (anchor_num, 1)).flatten()

    # (15,15) to (225,1) to (5,225) to (225x5,1)
    anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
    return anchor
