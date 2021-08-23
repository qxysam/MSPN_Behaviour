# MSPN- Beahavior_Prediction


## Repo Structure
This repo is organized as following:
```
$MSPN_HOME
|-- cvpack
|-- Beha_Train
|-- dataset
|   |-- COCO
|   |   |-- det_json
|   |   |-- gt_json
|   |   |-- images
|   |       |-- train2014
|   |       |-- val2014
|   |   |-- behavior_Annotations
|   |-- MPII
|       |-- det_json
|       |-- gt_json
|       |-- images
|   
|-- lib
|   |-- models
|   |-- utils
|
|
|-- model_logs
|
|-- README.md
|-- requirements.txt
```

1. Install Pytorch.
2. Install requirements:
 ```
 pip3 install -r requirements.txt
 ```
3. Install COCOAPI referring to [cocoapi website][3], or:
 ```
 git clone https://github.com/cocodataset/cocoapi.git $MSPN_HOME/lib/COCOAPI
 cd $MSPN_HOME/lib/COCOAPI/PythonAPI
 make install
 ```
 or: pip3 uninstall pycocotools
 
### Dataset

#### COCO

 Download images from [COCO website], and put train2014/val2014 splits into **$MSPN_HOME/dataset/COCO/images/** respectively.

#### MPII

 Download images from [MPII website], and put images into **$MSPN_HOME/dataset/MPII/images/**.


### Log
Create a directory to save logs and models:
```
mkdir $MSPN_HOME/model_logs
```

### Train
在根目录下（$MSPN_HOME）
and run:
```
python config.py -log
python -m torch.distributed.launch --nproc_per_node=gpu_num train.py
```
the ***gpu_num*** is the number of gpus.

### Test
```
python -m torch.distributed.launch --nproc_per_node=gpu_num test.py -i iter_num
```
the ***gpu_num*** is the number of gpus, and ***iter_num*** is the iteration number you want to test.


### Procedures AND Problems
1、Here, y = {0,1,2,3,4,5,6,7,8} is used to represent human behavior: 9 states: sleeping, walking, running, jumping, squatting, falling, sitting down, conflict, smoking (smoking and diet are very close, predicting smoking alone takes too much time, so they are combined);
2、Labeling the behavior corresponding to joint_points will add a behavior label to each human body on the coco data set;
3、Frame detection is carried out for real-time video. In case of sleeping, falling, conflict and smoking, an alarm will be given;
4、In addition to joint point detection, a new SVM network (MLP is also feasible, three layers, input, hidden and output) will be built as a multi classification network as a behavior judgment network;
5、In order to improve the generalization ability of the model, the data of 1-3 joint_points will be 0 randomly, and the joint points that are not predicted will also be replaced with 0 then the changed data will be added to the dataset;
6、In actual operation, it will not be detected every frame, and the performance cannot keep up. It is predicted once per second or every 2 seconds.
Summary:
          1、Falling is not easy to detect because the occurrence interval is very short. If you can detect every frame, it should be no problem. However, due to performance problems, We can't do it at present, but fortunately, falling will hardly happen.
          2、The transformation using the model optimization tensorrt method is unsuccessful. There are too many problems, incompatibilities and errors. If the second network is SVM method, it can not be optimized by deep learning method.



