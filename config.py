# encoding: utf-8
"""
@author: QQ
"""

import os, getpass
import os.path as osp
import argparse

# 替换成你的MSPN文件夹所在目录，此处视为项目的根目录
os.environ['MSPN_HOME'] = '/home/smart/MSPN/'

from easydict import EasyDict as edict
from dataset.attribute import load_dataset
from cvpack.utils.pyt_utils import ensure_dir


class Config:
    # -------- Directoy Config -------- #
    USER = getpass.getuser() #计算机用户名. Eg: smart
    ROOT_DIR = os.environ['MSPN_HOME'] #项目的根目录
    #在'MSPN/model_logs'文件夹下创建SMART/MSPN的文件夹，这里会存储日志等文件
    OUTPUT_DIR = osp.join(ROOT_DIR, 'model_logs', USER,
            osp.split(osp.split(osp.realpath(__file__))[0])[1]) 
    TEST_DIR = osp.join(OUTPUT_DIR, 'test_dir') #新建test_dir文件夹，存储测试结果
    TENSORBOARD_DIR = osp.join(OUTPUT_DIR, 'tb_dir') #新建tb_dir文件夹，存储日志等文件

    # -------- Data Config -------- #
    DATALOADER = edict()
    DATALOADER.NUM_WORKERS = 4
    DATALOADER.ASPECT_RATIO_GROUPING = False
    DATALOADER.SIZE_DIVISIBILITY = 0
    # DATALOADER = {'NUM_WORKERS': 4, 'ASPECT_RATIO_GROUPING': False, 'SIZE_DIVISIBILITY': 0}

    DATASET = edict()
    DATASET.NAME = 'COCO'
    
    dataset = load_dataset(DATASET.NAME)
    # dataset:   <dataset.attribute.COCO at 0x7f0a2ba77048>
    
    DATASET.KEYPOINT = dataset.KEYPOINT
    '''
    DATASET = 
            {'NAME': 'COCO',
             'KEYPOINT': {'NUM': 17,
                          'FLIP_PAIRS': [[1, 2],
                                         [3, 4],
                                         [5, 6],
                                         [7, 8],
                                         [9, 10],
                                         [11, 12],
                                         [13, 14],
                                         [15, 16]],
                          'UPPER_BODY_IDS': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                          'LOWER_BODY_IDS': [11, 12, 13, 14, 15, 16],
                          'LOAD_MIN_NUM': 1}}
    '''

    INPUT = edict()
    INPUT.NORMALIZE = True
    INPUT.MEANS = [0.406, 0.456, 0.485] # bgr
    INPUT.STDS = [0.225, 0.224, 0.229]
    '''
    INPUT = 
        {'NORMALIZE': True,
         'MEANS': [0.406, 0.456, 0.485],
         'STDS': [0.225, 0.224, 0.229]}
    '''
    # edict will automatcally convert tuple to list, so ..
    INPUT_SHAPE = dataset.INPUT_SHAPE #(256, 192)
    OUTPUT_SHAPE = dataset.OUTPUT_SHAPE #(64, 48)

    # -------- Model Config -------- #
    MODEL = edict()

    MODEL.BACKBONE = 'Res-50'
    MODEL.UPSAMPLE_CHANNEL_NUM = 256
    MODEL.STAGE_NUM = 2
    MODEL.OUTPUT_NUM = DATASET.KEYPOINT.NUM #17

    MODEL.DEVICE = 'cuda'

    MODEL.WEIGHT = osp.join(ROOT_DIR, 'lib/models/resnet-50_rename.pth')
    
    '''
    MODEL =
        {'BACKBONE': 'Res-50',
         'UPSAMPLE_CHANNEL_NUM': 256,
         'STAGE_NUM': 2,
         'OUTPUT_NUM': 17,
         'DEVICE': 'cuda',
         'WEIGHT': '/home/smart/MSPN/lib/models/resnet-50_rename.pth'}
    '''

    # -------- Training Config -------- #
    SOLVER = edict()
    SOLVER.BASE_LR = 5e-4 
    SOLVER.CHECKPOINT_PERIOD = 2400 
    SOLVER.GAMMA = 0.5
    SOLVER.IMS_PER_GPU = 32
    SOLVER.MAX_ITER = 96000 
    SOLVER.MOMENTUM = 0.9
    SOLVER.OPTIMIZER = 'Adam'
    SOLVER.WARMUP_FACTOR = 0.1
    SOLVER.WARMUP_ITERS = 2400 
    SOLVER.WARMUP_METHOD = 'linear'
    SOLVER.WEIGHT_DECAY = 1e-5 
    SOLVER.WEIGHT_DECAY_BIAS = 0
    '''
    SOLVER = 
        {'BASE_LR': 0.0005,
         'CHECKPOINT_PERIOD': 2400,
         'GAMMA': 0.5,
         'IMS_PER_GPU': 32,
         'MAX_ITER': 96000,
         'MOMENTUM': 0.9,
         'OPTIMIZER': 'Adam',
         'WARMUP_FACTOR': 0.1,
         'WARMUP_ITERS': 2400,
         'WARMUP_METHOD': 'linear',
         'WEIGHT_DECAY': 1e-05,
         'WEIGHT_DECAY_BIAS': 0}
    '''
    

    LOSS = edict()
    LOSS.OHKM = True
    LOSS.TOPK = 8
    LOSS.COARSE_TO_FINE = True
    # loss = {'OHKM': True, 'TOPK': 8, 'COARSE_TO_FINE': True}
    # OHEM: https://ranmaosong.github.io/2019/07/20/cv-imbalance-between-easy-and-hard-examples/

    RUN_EFFICIENT = False 
    # -------- Test Config -------- #
    TEST = dataset.TEST
    TEST.IMS_PER_GPU = 32
    '''
    TEST = 
        {'FLIP': True,
         'X_EXTENTION': 0.09,
         'Y_EXTENTION': 0.135,
         'SHIFT_RATIOS': [0.25],
         'GAUSSIAN_KERNEL': 5,
         'IMS_PER_GPU': 32}
    '''


config = Config()
cfg = config


def link_log_dir():
    '''
    再根目录下，建立一个log快捷文件夹，指向日志所在目录
    '''
    if not osp.exists('./log'):
        ensure_dir(config.OUTPUT_DIR)
        cmd = 'ln -s ' + config.OUTPUT_DIR + ' log'
        os.system(cmd)


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '-log', '--linklog', default=False, action='store_true')

    return parser


if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    if args.linklog:
        link_log_dir()
