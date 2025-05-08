import argparse
import json
import os
import os.path as osp
import sys

import cv2
import numpy as np

sys.path.append('..')

from ui import PyTable
from utils import read_list_from_file
from datasets import NUSCENES_ROOT
from tqdm import trange, tqdm

# 全局变量：保持代码原有风格，利用全局变量传递信息
gt_depth = None
pred_depth = None
current_pred_batch = None  # 当前批次的预测文件信息（例如 checkpoint_epoch=0）

def compute_metrics(pred, gt):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    result = {'abs_rel': abs_rel, 'sq_rel': sq_rel, 'rmse': rmse, 'rmse_log': rmse_log, 'a1': a1, 'a2': a2, 'a3': a3}
    return result

def print_table(title, data, str_format):
    table = PyTable(list(data.keys()), title)
    table.add_item({k: str_format.format(v) for k, v in data.items()})
    table.print_table()

def evaluate():
    # 尽量保持原有代码结构
    global pred_depth, gt_depth, current_pred_batch, args

    # 检查预测和真值的样本数量是否一致
    pred_len, gt_len = len(pred_depth), len(gt_depth)
    assert pred_len == gt_len, 'The length of predictions must be same as ground truth.'

    # 存储各项指标
    errors = {'abs_rel': [], 'sq_rel': [], 'rmse': [], 'rmse_log': [], 'a1': [], 'a2': [], 'a3': []}
    # 计算各个样本的评测结果
    for i in trange(gt_len, desc=f"评测 {current_pred_batch}"):
        pred, gt = pred_depth[i], gt_depth[i]
        mask = (gt > args.min_depth) & (gt < args.max_depth)
        # 调整尺寸
        gt_h, gt_w = gt.shape
        pred = cv2.resize(pred, (gt_w, gt_h), interpolation=cv2.INTER_NEAREST)
        # 取有效区域的数值
        pred_vals, gt_vals = pred[mask], gt[mask]
        if gt_vals.size <= 0:
            raise ValueError('The size of ground truth is zero.')
        # 按中位数比例缩放预测结果
        scale = np.median(gt_vals) / np.median(pred_vals)
        pred_vals *= scale
        pred_vals = np.clip(pred_vals, args.min_depth, args.max_depth)
        error = compute_metrics(pred_vals, gt_vals)
        for k in errors:
            errors[k].append(error[k])
    # 计算均值
    errors = {k: np.mean(v).item() for k, v in errors.items()}

    tqdm.write('Done.')
    print_table('Evaluation Result', errors, '{:.3f}')

    # 保存结果到 txt 文件中（追加写入，并标明当前批次）
    os.makedirs(osp.dirname(args.output_file_name), exist_ok=True)
    with open(args.output_file_name, 'a', encoding='utf-8') as fo:
        fo.write(f"===== 批次: {current_pred_batch} =====\n")
        json.dump(errors, fo)
        fo.write("\n\n")



def read_gt():
    """
    读取真值深度数据，依据命令行参数 weather 决定使用白天或夜晚的 npz 文件。
    npz 文件中的 key 顺序应与 split.txt 文件中样本名称顺序一致，
    本函数按照 split.txt 中的顺序提取对应的深度数组，并堆叠成一个 numpy 数组。
    """
    global args
    # 根据天气参数选择相应的 npz 文件
    gt_file = 'nuscenes_day_gt.npz' if args.weather == 'day' else 'nuscenes_night_gt.npz'
    gt_path = osp.join(NUSCENES_ROOT['test_gt'], gt_file)
    npz_data = np.load(gt_path)
    result = npz_data['depth']
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weather', type=str, help='Weather.', choices=['day', 'night'])
    parser.add_argument('--pred_dir', type=str, default='ns_result/', help='Directory where predictions stored.')
    parser.add_argument('--max_depth', type=float, default=60.0, help='Maximum depth value.')
    parser.add_argument('--min_depth', type=float, default=1e-5, help='Minimum depth value.')
    parser.add_argument('--output_file_name', type=str, default='/ns_result/ns_ckpt_err.txt', help='File name for saving result.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    # 示例中固定为夜晚，可根据需要修改
    args.weather = 'night'
    assert 0.0 < args.min_depth < 1.0, "min_depth 应在 (0,1) 之间"

    # 读取全部真值（从压缩的 .npz 文件中读取）
    gt_depth = read_gt()

    # 批量处理 pred_dir 下所有 .npy 文件
    npy_files = [f for f in os.listdir(args.pred_dir) if f.endswith('.npy')]
    # 按照批次中的 epoch 数字进行排序，例如从 checkpoint_epoch=0_predictions.npy 中提取数字 0
    npy_files.sort(key=lambda f: int(f.split('=')[1].split('_')[0]))
    # 如果存在输出文件则先清空
    if args.output_file_name is not None and osp.exists(args.output_file_name):
        os.remove(args.output_file_name)

    for f in npy_files:
        # 从文件名 checkpoint_epoch=0_predictions.npy 中提取批次信息，例如 checkpoint_epoch=0
        current_pred_batch = f.split('_predictions.npy')[0]
        pred_path = osp.join(args.pred_dir, f)
        pred_depth = np.load(pred_path)
        evaluate()

    print("所有批次评测完毕。指标结果已写入:", args.output_file_name)
