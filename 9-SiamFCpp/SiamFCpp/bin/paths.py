r"""
Get root path & root config path & 
"""
#import os.path as osp
import os
import sys  # isort:skip

module_name = "bin"
p = __file__
while os.path.basename(p) != module_name:
    p = os.path.dirname(p)
# video_analyst root
ROOT_PATH = os.path.dirname(p)
ROOT_CFG = os.path.join(ROOT_PATH, 'config.yaml')
sys.path.insert(0, ROOT_PATH)  # isort:skip