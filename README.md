# DIDAF-Depth: A Dual-Path Interaction and Dual-Attention Fusion Network with Transformer-CNN Architecture for Nighttime Depth Estimation
Chao Wei, Dongmei Zhou, Shengbing Chen, Xingyang Li, Hongyu Yao, Hao Luo  
Chengdu University of Technology, School of Mechanical and Electrical Engineering, Chengdu, China
# Introduction
This is the official repository for DIDAF-Depth: A Dual-Path Interaction and Dual-Attention Fusion Network with Transformer-CNN Architecture for Nighttime Depth Estimation. Since the paper is being submitted, we will continue to publish our training code and training results.
<p align="center">
  <img src="assets/Figure_3.jpg"
       alt="Overall architecture of DIDAF-Depth: TC-VIF encoder with dual-path (CNN & Transformer) and VIF modules; DCAFM for dual-attention fusion at four scales; EDCMNet decoder with PPM and EUCB; multi-scale depth heads; self-supervised training with PoseNet via photometric reconstruction."
       width="95%">
</p>
<p align="center"><em>Figure 3. Overall pipeline of DIDAF-Depth.</em></p>
# Dependency
- torch>=2.0.0
- mmcv-full==1.7.2
- mmengine>=0.10.5
- tqdm>=4.53
- pytorch-lightning>=2.0.4
- scikit-learn>=1.2.2
- torchmetrics>=1.0.2
# Datasets
For data preparation of RobotCar-Night and nuScenes-Night datasets, we follow the dataset used in "Regularizing Nighttime Weirdness: Efficient Self-supervised Monocular Depth Estimation in the Dark" [[pdf](https://arxiv.org/abs/2108.03830), [github](https://github.com/w2kun/RNW)]. Our work is based on [RobotCar](https://robotcar-dataset.robots.ox.ac.uk) and [nuScenes](https://www.nuscenes.org/nuscenes). <kbd>(2014-12-16-18-44-24, 2014-12-09-13-21-02)</kbd> of RobotCar and <kbd>(Package 01, 02, 05, 09, 10)</kbd> of nuScenes is enough for running the code. You can use the above official toolboxes produce the ground truth depth.  
After preparing datasets, we strongly recommend you to organize the directory structure as follows. The split files are provided in <kbd>split_files/</kbd>.
```
RobotCar-Night root directory
|__Package name (e.g. 2014-12-16-18-44-24)
  |__rgb (to store the .png color images)
    |__color image files
|__intrinsic.npy (to store the camera intrinsics)
|__test_split.txt (to store the test samples)
|__train_split.txt (to store the train samples)
|__STEPS_(day/night)_test_split.npz(day/night .npz ground truth depth maps)
```
```
nuScenes-Night root directory
|__sequences (to store sequence data)
   |__video clip number (e.g. scene-0004)
      |__file_list.txt (to store the image file names in this video clip)
      |__intrinsic.npy (to store the camera intrinsic of this video clip)
      |__image files described in file_list.txt
|__splits (to store split files)
   |__split files with name (day/night)_(train/test)_split.txt
|__test
   |__color (to store color images for testing)
   |__gt (to store ground truth depth maps w.r.t color)
```
__Note:__ You also need to configure the dataset path in <kbd>datasets/common.py</kbd>. 
# Training
Our model is trained using Distributed Data Parallel supported by [Pytorch-Lightning](https://github.com/Lightning-AI/pytorch-lightning). You can configure the training information in <kbd>train.py</kbd>, such as the training set, the number of GPUs, and the random seeds.
# Test
To test on RobotCar-Night or nuScenes-Night, you can run <kbd>test_(nuscenes/robotcar)_all.py</kbd>.
1.The best test results of our method in RobotCar-Night(maximum depth of 60m)
```
+---------------------------------------------------------------------------+
|                             Evaluation Result                             |
+---------------------------------------------------------------------------+
|  abs_rel  |  sq_rel  |   rmse  |  rmse_log  |    a1   |    a2   |    a3   |
+---------------------------------------------------------------------------+
|   0.116   |  0.850   |  3.962  |   0.174    |  0.876  |  0.958  |  0.984  |
+---------------------------------------------------------------------------+
```
```
1.The best test results of our method in nuScenes-Night(maximum depth of 60m)
+---------------------------------------------------------------------------+
|                             Evaluation Result                             |
+---------------------------------------------------------------------------+
|  abs_rel  |  sq_rel  |   rmse  |  rmse_log  |    a1   |    a2   |    a3   |
+---------------------------------------------------------------------------+
|  0.277    |  3.140   |  8.904  |   0.368    |  0.589  |  0.818  |  0.913  |
+---------------------------------------------------------------------------+
```
# Acknowledgement
We would like to thank the reviewers for their constructive comments and the authors of [RNW](https://arxiv.org/abs/2108.03830) and [STEPS](https://arxiv.org/abs/2302.01334) for their help and suggestions.   
If you are interested in our research, please keep following us and we will keep updating my results.
