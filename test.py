"""
@author: QQ
"""


import os
import argparse
from tqdm import tqdm
import numpy as np
import cv2
import json

import torch
import torch.distributed as dist

from cvpack.utils.logger import get_logger

from config import cfg
from network import MSPN
from lib.utils.dataloader import get_test_loader
from lib.utils.comm import is_main_process, synchronize, all_gather
from lib.utils.transforms import flip_back
from Behaviour import beha


def get_results(outputs, centers, scales, kernel=11, shifts=[0.25]):
    scales *= 200
    nr_img = outputs.shape[0] #32
     
    preds = np.zeros((nr_img, cfg.DATASET.KEYPOINT.NUM, 2)) #(32, 17, 2)
    maxvals = np.zeros((nr_img, cfg.DATASET.KEYPOINT.NUM, 1)) #(32, 17, 1)
    for i in range(nr_img): #针对每张图像
        score_map = outputs[i].copy()
        # print(score_map.shape) --> (17, 64, 48)
        score_map = score_map / 255 + 0.5
        kps = np.zeros((cfg.DATASET.KEYPOINT.NUM, 2)) # (17, 2)
        scores = np.zeros((cfg.DATASET.KEYPOINT.NUM, 1)) # (17, 1)
        border = 10
        dr = np.zeros((cfg.DATASET.KEYPOINT.NUM,
            cfg.OUTPUT_SHAPE[0] + 2 * border, cfg.OUTPUT_SHAPE[1] + 2 * border)) #(17, 64+20, 48+20)
        dr[:, border: -border, border: -border] = outputs[i].copy() #只在中间复制，border为0
        for w in range(cfg.DATASET.KEYPOINT.NUM):
            dr[w] = cv2.GaussianBlur(dr[w], (kernel, kernel), 0)  # 高斯模糊 https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html
        # print(dr.shape) --> (17, 84, 68)
        for w in range(cfg.DATASET.KEYPOINT.NUM): #针对每一个点
            for j in range(len(shifts)): # len(shifts) = 1 其实就执行一次，可以忽略
                if j == 0:
                    lb = dr[w].argmax() # argmax返回的是最大数的索引lb，找到点出现概率最大的地方的索引；注意这里是将一个二维数组[84,68]拉成一排，找相应的索引
                    y, x = np.unravel_index(lb, dr[w].shape) # 给定一个2维矩阵dr[w]，求第lb个元素的坐标是什么？
                    dr[w, y, x] = 0 #将点概率最大的地方置零
                    x -= border# 恢复，去掉边界
                    y -= border# 恢复，去掉边界

                lb = dr[w].argmax() # 相当于找点出现概率次大的地方，因为往前推3处，已经将最大可能性的地方置零
                py, px = np.unravel_index(lb, dr[w].shape)# 将概率次大的坐标返回
                dr[w, py, px] = 0 # 将概率次大的地方置零
                px -= border + x# 去掉边界之后，找概率次大与最大点横坐标的差（距离）
                py -= border + y# 去掉边界之后，找概率次大与最大点纵坐标的差（距离）
                ln = (px ** 2 + py ** 2) ** 0.5 #相当于检查概率次大与最大之间的距离，（px的平方+py的平方）开方
                
                if ln > 1e-3: #如果距离大于0.001，那么调整.为什么这么做？因为高斯kernel size生成的heatmap上是一个个的范围
                    x += shifts[j] * px / ln
                    y += shifts[j] * py / ln
            x = max(0, min(x, cfg.OUTPUT_SHAPE[1] - 1)) #max(0, min(x,48-1)) 意思是防止出圈
            y = max(0, min(y, cfg.OUTPUT_SHAPE[0] - 1)) #max(0, min(y,64-1)) 意思是防止出圈
            kps[w] = np.array([x * 4 + 2, y * 4 + 2]) #resize回去，原图INPUT_SHAPE是(256, 192)
            scores[w, 0] = score_map[w, int(round(y) + 1e-9), \
                    int(round(x) + 1e-9)] # + 1e-9防止为0 
        # aligned or not ...
        kps[:, 0] = kps[:, 0] / cfg.INPUT_SHAPE[1] * scales[i][0] + \
                centers[i][0] - scales[i][0] * 0.5
        kps[:, 1] = kps[:, 1] / cfg.INPUT_SHAPE[0] * scales[i][1] + \
                centers[i][1] - scales[i][1] * 0.5
        preds[i] = kps
        maxvals[i] = scores 
        
    return preds, maxvals


def compute_on_dataset(model, data_loader, device):
    model.eval()
    results = list() 
    cpu_device = torch.device("cpu")

    data = tqdm(data_loader) if is_main_process() else data_loader #read data

    for _, batch in enumerate(data):

        imgs, scores, centers, scales, img_ids = batch
        # 按照cfg里面，一次32张图像进去（batch size）进去
        # print(imgs.shape)-->torch.Size([32, 3, 256, 192])

        imgs = imgs.to(device)#加载GPU
        with torch.no_grad():
            outputs = model(imgs)#跑图
            outputs = outputs.to(cpu_device).numpy()
            # print(outputs.shape) --> (32, 17, 64, 48) (batch_size, #point, image_height, image_width)
            
            #这里cfg.TEST.FLIP为True，可以将其改为if cfg.TEST.FLIP == False加快速度
            #将测试图像翻转一下，再测试一遍，没啥必要其实，可以直接忽略
            if cfg.TEST.FLIP: 
                imgs_flipped = np.flip(imgs.to(cpu_device).numpy(), 3).copy()
                imgs_flipped = torch.from_numpy(imgs_flipped).to(device)
                outputs_flipped = model(imgs_flipped)
                outputs_flipped = outputs_flipped.to(cpu_device).numpy()
                outputs_flipped = flip_back(
                        outputs_flipped, cfg.DATASET.KEYPOINT.FLIP_PAIRS)
                outputs = (outputs + outputs_flipped) * 0.5

        centers = np.array(centers)
        scales = np.array(scales)
        #返回预测的点的坐标(32, 17, 2)与相应的分数(32, 17, 1)
        preds, maxvals = get_results(outputs, centers, scales,
                cfg.TEST.GAUSSIAN_KERNEL, cfg.TEST.SHIFT_RATIOS)#这里的GAUSSIAN_KERNEL为5，SHIFT_RATIOS为0.25
	    # 一张图17个算平均分
        kp_scores = maxvals.squeeze().mean(axis=1) #maxvals.shape = (32, 17, 1),maxvals.squeeze()去掉了维度变为(32,17)
        preds = np.concatenate((preds, maxvals), axis=2) # preds.shape=(32, 17, 3)前俩是坐标，最后是分数

        #将结果存成字典，以备之后将结果存为json文件
        for i in range(preds.shape[0]):
            keypoints = preds[i].reshape(-1).tolist()
           
            score = scores[i] * kp_scores[i]
            image_id = img_ids[i]

            results.append(dict(image_id=image_id,
                                category_id=1,
                                keypoints=keypoints,
                                score=score))

    return results


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu, logger):
    if is_main_process():
        logger.info("Accumulating ...")
    all_predictions = all_gather(predictions_per_gpu)

    if not is_main_process():
        return

    predictions = list()
    for p in all_predictions:
        predictions.extend(p)
    
    return predictions


def inference(model, data_loader, logger, device="cuda"):
    predictions = compute_on_dataset(model, data_loader, device)
    synchronize()
    predictions = _accumulate_predictions_from_multiple_gpus(
            predictions, logger)

    if not is_main_process():
        return

    return predictions    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--iter", "-i", type=int, default=-1)
    args = parser.parse_args()

    num_gpus = int(
            os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed =  num_gpus > 1
    
    # 这个if不用管，distributed是False，不执行。因为我们只有一块GPU
    # 这个if是分配多GPU一起工作的
    if distributed:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    # print(is_main_process())  #True 意思是只在主线程上处理，# False意思是准备多核并行处理


    if is_main_process() and not os.path.exists(cfg.TEST_DIR):
        os.mkdir(cfg.TEST_DIR)
    logger = get_logger(
            cfg.DATASET.NAME, cfg.TEST_DIR, args.local_rank, 'test_log.txt')

    if args.iter == -1:
        logger.info("Please designate one iteration.") #指定多少次迭代，单独跑图而不跑性能的话一次迭代就够了。

    model = MSPN(cfg) #加载模型
    device = torch.device(cfg.MODEL.DEVICE) #我们依然还是用GPU来训练
    model.to(cfg.MODEL.DEVICE) #将model加载入GPU

    #model_file = os.path.join(cfg.OUTPUT_DIR, "iter-{}.pth".format(args.iter)) #如果你想直接调用自己训练的第i次iteration模型，-i 次数
    model_file = '/home/smart/MSPN/lib/models/mspn_2xstg_coco.pth'#或者直接更改模型所在的绝对路径。

    #载入预训练模型
    if os.path.exists(model_file):
        state_dict = torch.load(
                model_file, map_location=lambda storage, loc: storage)
        state_dict = state_dict['model']
        model.load_state_dict(state_dict)

    data_loader = get_test_loader(cfg, num_gpus, args.local_rank, 'val',
            is_dist=distributed)#加载测试数据 <torch.utils.data.dataloader.DataLoader object at 0x7ff224f5c0b8>

    results = inference(model, data_loader, logger, device)
    synchronize() #没用，因为我们不是多GPU训练
    results=beha(results)


    # is_main_process == True; 下面这部分是将结果写成Json
    if is_main_process():
        logger.info("Dumping results ...")
        results.sort(
                key=lambda res:(res['image_id'], res['score']), reverse=True) 
        results_path = os.path.join(cfg.TEST_DIR, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f)
        logger.info("Get all results.")

        data_loader.ori_dataset.evaluate(results_path)


if __name__ == '__main__':
    main()
