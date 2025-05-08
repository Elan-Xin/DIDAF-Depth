import argparse
import json
import os
import sys
import glob
import cv2
import numpy as np

sys.path.append('..')

from tqdm import trange, tqdm
from ui import PyTable
from utils import read_list_from_file
from transforms import CenterCrop
from datasets import ROBOTCAR_ROOT

# 目标大小
_TARGET_SIZE = (1152, 640)


def compute_metrics(pred, gt):
    """
    计算评估指标
    :param pred: 预测深度图
    :param gt: 地面真实深度图
    :return: 评估指标（如abs_rel, rmse等）
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()  # 比较误差小于1.25的比例
    a2 = (thresh < 1.25 ** 2).mean()  # 比较误差小于1.25^2的比例
    a3 = (thresh < 1.25 ** 3).mean()  # 比较误差小于1.25^3的比例

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())  # 均方根误差（RMSE）
    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())  # 对数均方根误差（RMSE_log）

    abs_rel = np.mean(np.abs(gt - pred) / gt)  # 平均相对误差
    sq_rel = np.mean(((gt - pred) ** 2) / gt)  # 平均平方相对误差

    result = {'abs_rel': abs_rel, 'sq_rel': sq_rel, 'rmse': rmse, 'rmse_log': rmse_log, 'a1': a1, 'a2': a2, 'a3': a3}
    return result


def read_gt(dn, fs):
    """
    读取给定目录下的地面真实深度图
    :param dn: 目录路径
    :param fs: 文件名列表
    :return: 真实深度图列表
    """
    result = []
    for f in fs:
        result.append(np.load(os.path.join(dn, '{}.npy'.format(f))))
    return result


def print_table(title, data, str_format):
    """
    打印结果表格
    :param title: 表格标题
    :param data: 数据字典
    :param str_format: 数值格式化字符串
    """
    table = PyTable(list(data.keys()), title)
    table.add_item({k: str_format.format(v) for k, v in data.items()})
    table.print_table()


def evaluate(pred_depth, gt_depth):
    """
    评估预测结果与地面真实深度图之间的指标差异
    :param pred_depth: 预测深度图列表
    :param gt_depth: 真实深度图列表
    :return: 错误指标字典
    """
    # 检查预测和真实深度图数量是否一致
    pred_len, gt_len = len(pred_depth), len(gt_depth)
    assert pred_len == gt_len, '预测结果与地面真实深度图数量不一致。'

    # 存储结果
    errors = {'abs_rel': [], 'sq_rel': [], 'rmse': [], 'rmse_log': [], 'a1': [], 'a2': [], 'a3': []}

    # 初始化裁剪变换
    crop = CenterCrop(*_TARGET_SIZE)

    # 计算每个图像的误差
    for i in trange(gt_len):
        # 获取预测深度图和真实深度图
        pred, gt = pred_depth[i], gt_depth[i]
        gt = crop(gt, inplace=False)  # 对真实深度图进行裁剪
        mask = (gt > args.min_depth) & (gt < args.max_depth)  # 掩膜，剔除无效的深度值

        # 调整预测深度图尺寸与真实深度图一致
        gt_h, gt_w = gt.shape
        # print( gt_h, gt_w, 1111111111111)
        pred = cv2.resize(pred, (gt_w, gt_h), interpolation=cv2.INTER_NEAREST)

        # 获取掩膜下的值
        pred_vals, gt_vals = pred[mask], gt[mask]

        # 如果真实值为空，抛出异常
        if gt_vals.size <= 0:
            raise ValueError('地面真实深度图中有效值为空。')

        # 计算尺度因子（根据真实深度图的中位数与预测深度图的中位数）
        scale = np.median(gt_vals) / np.median(pred_vals)
        pred_vals *= scale  # 根据尺度因子调整预测值
        pred_vals = np.clip(pred_vals, args.min_depth, args.max_depth)  # 限制预测深度图值的范围

        # 计算各项评估指标
        error = compute_metrics(pred_vals, gt_vals)

        # 将每个指标添加到结果中
        for k in errors:
            errors[k].append(error[k])

    # 计算每个指标的平均值
    errors = {k: np.mean(v).item() for k, v in errors.items()}

    # 打印计算完成的提示
    tqdm.write('计算完成。')
    # output
    print_table('Evaluation Result', errors, '{:.3f}')

    # 返回评估结果
    return errors


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='night', help='数据集根目录.', choices=['day', 'night'])
    parser.add_argument('--pred_dir', type=str, default='rc_result/', help='预测结果所在目录.')
    parser.add_argument('--max_depth', type=float, default=60.0, help='最大深度值.')
    parser.add_argument('--min_depth', type=float, default=1e-5, help='最小深度值.')
    parser.add_argument('--output_file_name', type=str, default='./rc_result/rc_result.txt', help='保存结果的文件路径.')
    return parser.parse_args()

def save_results(errors, output_file_name, epoch_num):
    """
    将评估结果保存到文件
    :param errors: 评估结果字典
    :param output_file_name: 输出文件名
    :param epoch_num: 当前epoch编号
    """
    # 确保输出文件的目录存在，如果不存在则创建
    output_dir = os.path.dirname(output_file_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 如果文件不存在，则创建并写入头部
    file_exists = os.path.exists(output_file_name)
    with open(output_file_name, 'a') as f:
        # 如果文件不存在，写入文件头部信息
        if not file_exists:
            header = {
                "epoch": "Epoch Number",
                "abs_rel": "Absolute Relative Error",
                "sq_rel": "Squared Relative Error",
                "rmse": "Root Mean Square Error",
                "rmse_log": "Logarithmic Root Mean Square Error",
                "a1": "A1 Accuracy",
                "a2": "A2 Accuracy",
                "a3": "A3 Accuracy"
            }
            f.write(json.dumps(header) + '\n')

        # 保存当前epoch的评估结果
        result = {"epoch": epoch_num, **errors}
        f.write(json.dumps(result) + '\n')
    print(f"结果已保存至 {output_file_name}_epoch_{epoch_num}")


if __name__ == '__main__':
    # 解析命令行参数
    args = parse_args()
    args.data_root = 'night'  # 设置数据集类型
    root_dir = r'G:\monodepth\datasets\robotcar\all\image_data\night\2014-12-16-18-44-24'
    args.pred_dir = r"G:\contrast\rc\steps1"
    args.output_file_name = './rc_result/rc_result_base.txt'
    # 根据数据集类型加载真实深度图路径
    if args.data_root == 'night':
        gt_path = os.path.join(root_dir, "STEPS_night_test_split.npz")
    elif args.data_root == 'day':
        gt_path = os.path.join(root_dir, "STEPS_daytime_test_split.npz")

    # 加载地面真实深度图
    gt_depth = np.load(gt_path, fix_imports=True, encoding='latin1', allow_pickle=True)['depth_images_array']

    # 获取预测结果文件列表
    # checkpoint_files = args.pred_dir
    checkpoint_dir = args.pred_dir
    checkpoint_files = glob.glob(
        os.path.join(checkpoint_dir, 'swin_comer_resnet50_DG_predictions_night_epoch29.npy'.format(args.data_root))
        # os.path.join(checkpoint_dir, 'swin_comer_resnet50_predictions_{}.npy'.format(args.data_root))
    )
    # print(checkpoint_files)
    # # 按照 epoch 数字大小排序
    # checkpoint_files = sorted(
    #     checkpoint_files,
    #     key=lambda x: int(os.path.basename(x).split('epoch')[1].split('.npy')[0])
    # )

    # 遍历每个批次（epoch）的预测文件，计算对应的评估指标
    for checkpoint_file in checkpoint_files:
        # epoch_num = os.path.basename(checkpoint_file).split('epoch')[1].split('.npy')[0]
        # print(f"正在评估 epoch {epoch_num} 的预测结果...")

        # 加载当前epoch的预测结果
        pred_depth = np.load(checkpoint_file)

        # 评估当前预测结果并获取指标
        errors = evaluate(pred_depth, gt_depth)

        # 如果需要，保存评估结果到文件
        if args.output_file_name:
            args.output_file_name = './rc_result/rc_{}_result.txt'.format(args.data_root)
            # 保存评估结果到文件
            save_results(errors, args.output_file_name, 29)
