import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from ultralytics import YOLO
import torch
import random
import numpy as np

# ----------------------------
# 固定随机种子（确保可复现）
# ----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    # 设置随机种子
    set_seed(42)

    # ----------------------------
    # 可修改区域（你的数据配置）D:\INTROAI\raw data\Walk Assist.v3i.yolov8
    # ----------------------------
    DATA_YAML = r"D:\INTROAI\raw data\Walk Assist.v3i.yolov8\data.yaml" # yaml 文件名
    MODEL = "yolov8n.pt"                    # 可换 yolov8s.pt、yolov8m.pt
    PROJECT = "runs_blindroad"              # 实验存放目录
    NAME = "yv8n_merged_v"                 # 实验名称（用于复现实验）
    DEVICE = 0                              # GPU id，CPU 就写 "cpu"

    # ----------------------------
    # 初始化模型
    # ----------------------------
    model = YOLO(MODEL)

    # ----------------------------
    # 训练（可复现 + 可调参）
    # ----------------------------
    model.train(
        data=DATA_YAML,
        epochs=150,
        imgsz=640,
        batch=4,
        device=DEVICE,
        seed=42,
        project=PROJECT,
        name=NAME,

        # --- 调优关键参数 ---
        lr0=0.0008,            # 初始学习率
        lrf=0.01,            # 最终学习率比例
        weight_decay=0.0005, # L2 正则
        patience=20,         # 早停 patience
        warmup_epochs=3,     # ✅ 小数据做一个短 warmup，防止一开始震荡太大
        close_mosaic=10,     # ✅ 最后 10 个 epoch 关 mosaic 做微调

        # --- 额外参数 ---
        save_period=10,      # 每 10 轮保存一次
        verbose=True,        # 打印详细信息
        workers=0,           # 单进程加载数据
    )


if __name__ == "__main__":
    main()
