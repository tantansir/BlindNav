#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
yolo_data_balance.py

工具脚本：
1) oversample-rare: 对包含稀有类别的样本做离线过采样，提升少数类在训练集中的比例
2) mine-hard: 在第一阶段训练后，从验证集中挖掘难例并复制到训练集，用于第二阶段微调

假设数据目录结构为标准 YOLOv8：
data_root/
  images/
    train/
    val/
  labels/
    train/
    val/

预测结果假设是通过 YOLO 的 val 或 detect 命令导出的 txt 文件：
pred_labels/
  xxx.txt    # YOLO 格式: class x_center y_center w h (归一化)

使用示例：

1) 对类别计数少于 500 的类别做过采样（复制 3 倍）：
python yolo_data_balance.py oversample-rare \
    --data-root D:/INTROAI/raw_data/blind_merged_v1 \
    --min-count 500 \
    --dup-factor 3

2) 从验证集中挖掘难例（图像级 recall < 0.9）并复制 2 倍到训练集：
python yolo_data_balance.py mine-hard \
    --data-root D:/INTROAI/raw_data/blind_merged_v1 \
    --val-pred-dir D:/INTROAI/runs_blindroad/yv8n_merged_v1/val/labels \
    --recall-threshold 0.9 \
    --dup-factor 2
"""

import argparse
import shutil
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import math


# ---------------------------
# 工具函数
# ---------------------------

def read_yolo_labels(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """
    读取 YOLO txt 标签文件，返回 [(cls, xc, yc, w, h), ...]
    如果文件不存在或为空，返回空列表。
    """
    if not label_path.exists():
        return []
    boxes = []
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            xc, yc, w, h = map(float, parts[1:])
            boxes.append((cls, xc, yc, w, h))
    return boxes


def iou_yolo_box(box1, box2) -> float:
    """
    计算两个 YOLO 格式 box 的 IoU（假设坐标已经是归一化的 [0,1]）。
    box = (cls, xc, yc, w, h)
    """
    _, x1, y1, w1, h1 = box1
    _, x2, y2, w2, h2 = box2

    x1_min = x1 - w1 / 2
    x1_max = x1 + w1 / 2
    y1_min = y1 - h1 / 2
    y1_max = y1 + h1 / 2

    x2_min = x2 - w2 / 2
    x2_max = x2 + w2 / 2
    y2_min = y2 - h2 / 2
    y2_max = y2 + h2 / 2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_w = max(0.0, inter_x_max - inter_x_min)
    inter_h = max(0.0, inter_y_max - inter_y_min)
    inter_area = inter_w * inter_h

    area1 = w1 * h1
    area2 = w2 * h2

    union = area1 + area2 - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


# ---------------------------
# 1. 类别重加权采样（离线过采样）
# ---------------------------

def oversample_rare_classes(
    data_root: Path,
    min_count: int,
    dup_factor: int,
    rare_class_ids: List[int] = None,
) -> None:
    """
    扫描 labels/train 统计类别频次；对样本稀少的类别做离线过采样。
    如果 rare_class_ids 为 None，则自动将计数 < min_count 的类别视为少数类。

    过采样方式：将包含少数类的 image + label 复制 dup_factor 次，
    文件名后加 _dup1, _dup2, ...
    """
    labels_train = data_root / "labels" / "train"
    images_train = data_root / "images" / "train"

    assert labels_train.exists() and images_train.exists(), "train 路径不存在，请检查 data_root"

    print(f"[INFO] Scanning train labels under {labels_train} ...")
    class_counter: Counter = Counter()
    image_to_classes: Dict[Path, set] = {}

    for label_file in labels_train.glob("*.txt"):
        boxes = read_yolo_labels(label_file)
        cls_set = set()
        for cls, *_ in boxes:
            class_counter[cls] += 1
            cls_set.add(cls)
        # 只记录有物体的图
        if cls_set:
            image_file = images_train / (label_file.stem + ".jpg")
            if not image_file.exists():
                image_file = images_train / (label_file.stem + ".png")
            if image_file.exists():
                image_to_classes[image_file] = cls_set

    print("[INFO] Class counts:")
    for c, cnt in sorted(class_counter.items()):
        print(f"  class {c}: {cnt}")

    if rare_class_ids is None:
        rare_class_ids = [c for c, cnt in class_counter.items() if cnt < min_count]
        print(f"[INFO] Auto detected rare classes (count < {min_count}): {rare_class_ids}")
    else:
        print(f"[INFO] Using user specified rare classes: {rare_class_ids}")

    if not rare_class_ids:
        print("[INFO] No rare classes found. Nothing to oversample.")
        return

    # 找出包含少数类的图片
    rare_images = []
    for img_path, cls_set in image_to_classes.items():
        if any(c in cls_set for c in rare_class_ids):
            rare_images.append(img_path)

    print(f"[INFO] Found {len(rare_images)} images containing rare classes.")

    # 复制这些图片及其 label
    for img_path in rare_images:
        label_path = labels_train / (img_path.stem + ".txt")
        for k in range(1, dup_factor + 1):
            new_img = img_path.with_name(f"{img_path.stem}_dup{k}{img_path.suffix}")
            new_label = label_path.with_name(f"{label_path.stem}_dup{k}{label_path.suffix}")
            if new_img.exists() or new_label.exists():
                continue
            shutil.copy2(img_path, new_img)
            shutil.copy2(label_path, new_label)

    print(f"[INFO] Oversampling done. Duplicated each rare-class image {dup_factor} times.")


# ---------------------------
# 2. 两阶段难例挖掘
# ---------------------------

def find_hard_examples(
    data_root: Path,
    val_pred_dir: Path,
    recall_threshold: float,
    iou_threshold: float = 0.5,
) -> List[Path]:
    """
    在验证集中挖掘难例。难例定义为：图像级 recall < recall_threshold。
    使用 YOLO txt 预测结果和 val 标签计算匹配情况。
    返回需要增强的 val 图像路径列表。
    """
    labels_val = data_root / "labels" / "val"
    images_val = data_root / "images" / "val"

    assert labels_val.exists() and images_val.exists(), "val 路径不存在"
    assert val_pred_dir.exists(), "预测结果路径不存在"

    hard_images: List[Path] = []

    print(f"[INFO] Mining hard examples from {labels_val} using predictions in {val_pred_dir} ...")

    for label_file in labels_val.glob("*.txt"):
        gt_boxes = read_yolo_labels(label_file)
        if not gt_boxes:
            continue

        pred_file = val_pred_dir / label_file.name
        pred_boxes = read_yolo_labels(pred_file)

        if not pred_boxes:
            # 没有任何预测，recall 为 0
            img_path = images_val / (label_file.stem + ".jpg")
            if not img_path.exists():
                img_path = images_val / (label_file.stem + ".png")
            if img_path.exists():
                hard_images.append(img_path)
            continue

        matched_gt = 0
        used_pred = set()

        for gt in gt_boxes:
            best_iou = 0.0
            best_idx = -1
            for idx, pred in enumerate(pred_boxes):
                if idx in used_pred:
                    continue
                if gt[0] != pred[0]:
                    continue
                iou = iou_yolo_box(gt, pred)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_iou >= iou_threshold:
                matched_gt += 1
                used_pred.add(best_idx)

        recall = matched_gt / len(gt_boxes)
        if recall < recall_threshold:
            img_path = images_val / (label_file.stem + ".jpg")
            if not img_path.exists():
                img_path = images_val / (label_file.stem + ".png")
            if img_path.exists():
                hard_images.append(img_path)

    print(f"[INFO] Found {len(hard_images)} hard images with recall < {recall_threshold}.")
    return hard_images


def duplicate_images_to_train(
    data_root: Path,
    image_paths: List[Path],
    dup_factor: int,
) -> None:
    """
    将给定图像及其 label 从 val 复制到 train，多复制 dup_factor 次。
    文件名加 _hard_dupX。
    """
    images_train = data_root / "images" / "train"
    labels_train = data_root / "labels" / "train"
    labels_val = data_root / "labels" / "val"

    for img_path in image_paths:
        stem = img_path.stem
        label_src = labels_val / f"{stem}.txt"
        if not label_src.exists():
            continue
        for k in range(1, dup_factor + 1):
            new_img = images_train / f"{stem}_hard_dup{k}{img_path.suffix}"
            new_label = labels_train / f"{stem}_hard_dup{k}.txt"
            if new_img.exists() or new_label.exists():
                continue
            shutil.copy2(img_path, new_img)
            shutil.copy2(label_src, new_label)

    print(f"[INFO] Duplicated {len(image_paths)} hard images x{dup_factor} into train set.")


# ---------------------------
# CLI
# ---------------------------

def main():
    parser = argparse.ArgumentParser(
        description="YOLOv8 data balancing tools: oversample rare classes and mine hard examples."
    )
    subparsers = parser.add_subparsers(dest="command")

    # oversample-rare
    p_over = subparsers.add_parser(
        "oversample-rare", help="offline oversampling for rare classes"
    )
    p_over.add_argument("--data-root", type=str, required=True,
                        help="root directory of YOLO dataset (contains images/ and labels/)")
    p_over.add_argument("--min-count", type=int, default=500,
                        help="classes with count < min-count are treated as rare")
    p_over.add_argument("--dup-factor", type=int, default=3,
                        help="duplicate times for each rare-class image")
    p_over.add_argument("--rare-classes", type=int, nargs="*",
                        help="optional explicit rare class ids, if given min-count is ignored")

    # mine-hard
    p_hard = subparsers.add_parser(
        "mine-hard", help="mine hard examples from val and copy into train"
    )
    p_hard.add_argument("--data-root", type=str, required=True,
                        help="root directory of YOLO dataset (contains images/ and labels/)")
    p_hard.add_argument("--val-pred-dir", type=str, required=True,
                        help="directory of YOLO predictions for val set (txt format)")
    p_hard.add_argument("--recall-threshold", type=float, default=0.9,
                        help="images with recall < threshold are treated as hard")
    p_hard.add_argument("--iou-threshold", type=float, default=0.5,
                        help="IoU threshold for matching gt and prediction")
    p_hard.add_argument("--dup-factor", type=int, default=2,
                        help="duplicate times for each hard image when copying to train")

    args = parser.parse_args()

    if args.command == "oversample-rare":
        data_root = Path(args.data_root)
        oversample_rare_classes(
            data_root=data_root,
            min_count=args.min_count,
            dup_factor=args.dup_factor,
            rare_class_ids=args.rare_classes,
        )
    elif args.command == "mine-hard":
        data_root = Path(args.data_root)
        val_pred_dir = Path(args.val_pred_dir)
        hard_images = find_hard_examples(
            data_root=data_root,
            val_pred_dir=val_pred_dir,
            recall_threshold=args.recall_threshold,
            iou_threshold=args.iou_threshold,
        )
        duplicate_images_to_train(
            data_root=data_root,
            image_paths=hard_images,
            dup_factor=args.dup_factor,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
