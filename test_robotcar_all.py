import argparse
import os
import os.path as osp
import glob

import cv2
import numpy as np
import pytorch_lightning
import torch
from mmengine.config import Config
from torchvision.transforms import ToTensor
from tqdm import tqdm

from datasets import ROBOTCAR_ROOT
from models import MODELS
from models.utils import disp_to_depth
from transforms import CenterCrop
from utils import read_list_from_file, save_color_disp, save_disp

# 裁剪尺寸
_CROP_SIZE = (1152, 640)
# 输出目录
_OUT_DIR = 'evaluation/rc_result/'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='night', help='Root directory of dataset.', choices=['day', 'night'])
    parser.add_argument('--config', type=str, default='steps_rc', help='配置文件。')
    parser.add_argument('--checkpoint', type=str, default='/root/lanyun-tmp/steps/checkpoints/didaf_rc/')
    parser.add_argument('--test', type=int, default=1, help='测试模式。')
    parser.add_argument('--vis', type=int, default=0, help='可视化标志。')
    return parser.parse_args()


if __name__ == '__main__':
    # 解析参数m
    args = parse_args()
    args.data_root = 'night'
    # 配置
    cfg = Config.fromfile(osp.join('configs/', f'{args.config}.yaml'))
    cfg.test = args.test
    # 打印信息
    print('正在使用 {} 进行评估...'.format(os.path.basename(args.config)))
    # 设备
    device = torch.device('cuda:0')
    # 读取列表文件
    root_dir = ROBOTCAR_ROOT[args.data_root] if args.data_root in ROBOTCAR_ROOT else args.data_root
    test_items = read_list_from_file(os.path.join(root_dir, 'test_split.txt'), 1)
    test_items = sorted(test_items)
    # 模型
    model_name = cfg.model.name
    net: pytorch_lightning.LightningModule = MODELS.build(name=model_name, cfg=cfg)
    net.to(device)
    net.eval()
    # 获取检查点文件列表
    checkpoint_dir = os.path.dirname(args.checkpoint)
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_epoch=*.ckpt'))
    checkpoint_files = sorted(checkpoint_files)
    # 变换
    crop = CenterCrop(*_CROP_SIZE)
    to_tensor = ToTensor()
    # 遍历每个检查点文件
    for checkpoint_file in checkpoint_files:
        # 提取 epoch 信息
        basename = os.path.basename(checkpoint_file)
        epoch_num = basename.split('=')[1].split('.')[0]
        # 加载模型权重
        net.load_state_dict(torch.load(checkpoint_file)['state_dict'])
        print('成功从检查点 {} 加载权重。'.format(checkpoint_file))
        # 存储结果
        predictions = []
        # 关闭梯度计算
        with torch.no_grad():
            # 预测
            for idx, item in enumerate(tqdm(test_items)):
                # 路径
                path = os.path.join(root_dir, 'rgb/stereo/left/', '{}.png'.format(item))
                # 读取图像
                rgb = cv2.imread(path)
                gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
                # 裁剪
                rgb = crop(rgb)
                gray = crop(gray)
                # 调整大小
                rgb = cv2.resize(rgb, (cfg.dataset['width'], cfg.dataset['height']), interpolation=cv2.INTER_LINEAR)
                gray = cv2.resize(gray, (cfg.dataset['width'], cfg.dataset['height']), interpolation=cv2.INTER_LINEAR)
                # 转为张量
                t_rgb = to_tensor(rgb).unsqueeze(0).to(device)
                t_gray = to_tensor(gray).unsqueeze(0).to(device)
                # 输入网络
                outputs = net({('color', 0, 0): t_rgb,
                               ('color_aug', 0, 0): t_rgb,
                               ('color_gray', 0, 0): t_gray})
                disp = outputs[("disp", 0, 0)]
                scaled_disp, depth = disp_to_depth(disp, 0.1, 100)
                depth = depth.cpu()[0, 0, :, :].numpy()
                # 添加到结果
                predictions.append(depth)
                # 可视化
                if args.vis:
                    scaled_disp = scaled_disp.cpu()[0, 0, :, :].numpy()
                    out_fn = os.path.join("./evaluation/rc_result/vis/{}".format(args.data_root), '{}_depth.png'.format("%05d" % idx))
                    color_fn = os.path.join("./evaluation/rc_result/vis/{}".format(args.data_root), '{}_rgb.png'.format("%05d" % idx))
                    # 确保目录存在
                    os.makedirs(os.path.dirname(out_fn), exist_ok=True)
                    os.makedirs(os.path.dirname(color_fn), exist_ok=True)
                    save_disp(rgb, scaled_disp, out_fn, color_fn, max_p=95, dpi=256)
        # 堆叠结果
        predictions = np.stack(predictions, axis=0)
        # 保存预测结果，文件名中加入 epoch 信息
        output_filename = 'swin_comer_resnet50_DG_predictions_{}_epoch{}.npy'.format(args.data_root, epoch_num)
        os.makedirs(_OUT_DIR, exist_ok=True)
        np.save(os.path.join(_OUT_DIR, output_filename), predictions, allow_pickle=False)
        # 显示信息
        print('已保存预测结果到 {}。'.format(output_filename))
    tqdm.write('完成。')
