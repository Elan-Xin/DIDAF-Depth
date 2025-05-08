#!/usr/bin/env python3
import argparse
import os
import os.path as osp
import cv2
import numpy as np
import pytorch_lightning
import torch
import time
from mmcv import Config
from torchvision.transforms import ToTensor
from tqdm import tqdm

from datasets import NUSCENES_ROOT
from models import MODELS
from models.utils import disp_to_depth
from utils import read_list_from_file, save_color_disp, save_disp


def parse_args():
    parser = argparse.ArgumentParser(description="一次性测试所有checkpoint")
    parser.add_argument('--root_dir', type=str, default='night',  help='测试数据集名称（对应split文件中的前缀，如 nuscenes）')
    parser.add_argument('--config', type=str, default='didaf_ns',  help='配置文件名称，不需要后缀.yaml')
    parser.add_argument('--checkpoint_dir', default='/root/lanyun-tmp/steps/checkpoints/didaf_ns/',type=str, help='存放所有checkpoint的文件夹路径')
    parser.add_argument('--test', type=int, default=1, help='测试模式参数')
    parser.add_argument('--vis', type=int, default=0, help='可视化标志，1为开启')
    parser.add_argument('--out_dir', type=str, default='evaluation/ns_result1/', help='存放所有测试结果的文件夹')
    return parser.parse_args()


def main():
    args = parse_args()

    # 加载配置文件
    cfg = Config.fromfile(osp.join('configs', f'{args.config}.yaml'))
    cfg.test = args.test

    # 创建输出目录
    if not osp.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # 获取所有checkpoint文件
    ckpt_list = sorted([
        osp.join(args.checkpoint_dir, f)
        for f in os.listdir(args.checkpoint_dir)
        if f.endswith('.pth') or f.endswith('.ckpt')
    ])
    if not ckpt_list:
        print("未找到任何checkpoint文件！")
        return

    # 读取测试数据列表
    split_file = osp.join(NUSCENES_ROOT['split'], f'{args.root_dir}_test_split.txt')
    test_items = read_list_from_file(split_file, 1)
    # test_items = sorted(test_items)
    print(f"共找到 {len(test_items)} 个测试样本。")

    # 如果需要可视化，则准备可视化文件夹（统一放在输出目录下）
    if args.vis:
        vis_dir = osp.join(args.out_dir, 'vis')
        if not osp.exists(vis_dir):
            os.makedirs(vis_dir)

    # 设备设定
    device = torch.device('cuda:0')
    to_tensor = ToTensor()

    # 遍历每个checkpoint进行测试
    for ckpt in ckpt_list:
        print("====================================")
        print(f"开始测试 checkpoint：{ckpt}")

        # 构建模型并加载checkpoint
        model_name = cfg.model.name
        net: pytorch_lightning.LightningModule = MODELS.build(name=model_name, cfg=cfg)
        state = torch.load(ckpt)
        net.load_state_dict(state['state_dict'])
        net.to(device)
        net.eval()
        print(f"成功加载权重：{ckpt}")

        predictions = []
        total_time = 0

        with torch.no_grad():
            for idx, item in enumerate(tqdm(test_items, desc=f"测试 {osp.basename(ckpt)}")):
                img_path = osp.join(NUSCENES_ROOT['test_color'], item + '.jpg')
                rgb = cv2.imread(img_path)
                if rgb is None:
                    print(f"无法读取图片：{img_path}")
                    continue
                gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
                rgb = cv2.resize(rgb, (cfg.dataset['width'], cfg.dataset['height']), interpolation=cv2.INTER_LINEAR)
                gray = cv2.resize(gray, (cfg.dataset['width'], cfg.dataset['height']), interpolation=cv2.INTER_LINEAR)
                t_rgb = to_tensor(rgb).unsqueeze(0).to(device)
                t_gray = to_tensor(gray).unsqueeze(0).to(device)
                start_time = time.time()
                outputs = net({('color', 0, 0): t_rgb,
                               ('color_aug', 0, 0): t_rgb,
                               ('color_gray', 0, 0): t_gray})
                disp = outputs[("disp", 0, 0)]
                scaled_disp, depth = disp_to_depth(disp, 0.1, 100)
                end_time = time.time()
                total_time += (end_time - start_time)
                depth_np = depth.cpu()[0, 0, :, :].numpy()
                predictions.append(depth_np)

                # 如果开启可视化，则保存可视化图片
                if args.vis:
                    scaled_disp_np = scaled_disp.cpu()[0, 0, :, :].numpy()
                    vis_depth_path = osp.join(args.out_dir, f'{osp.basename(ckpt).split(".")[0]}_{idx:03d}_depth.png')
                    vis_rgb_path = osp.join(args.out_dir, f'{osp.basename(ckpt).split(".")[0]}_{idx:03d}_rgb.png')
                    save_color_disp(rgb[:, :, ::-1], scaled_disp_np, vis_depth_path, max_p=95, dpi=256)
                    save_disp(rgb[:, :, ::-1], scaled_disp_np, vis_depth_path, vis_rgb_path, max_p=95, dpi=256)

        # 计算 FPS
        fps = int(len(test_items) / total_time) if total_time > 0 else 0
        print("====================================")
        print(f"checkpoint：{ckpt} 测试完成，平均FPS：{fps}")

        # 保存预测结果，文件名中包含当前checkpoint的名称
        out_file = osp.join(args.out_dir, f'{osp.basename(ckpt).split(".")[0]}_predictions.npy')
        np.save(out_file, np.stack(predictions, axis=0), allow_pickle=False)
        print(f"预测结果已保存到：{out_file}")

    print("所有checkpoint测试完毕。")


if __name__ == '__main__':
    main()
