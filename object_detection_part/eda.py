"""
EDA for YOLO Dataset (blind_merged.v2)

功能：
1. 统计每张图片的标注数量分布，并画直方图（附带类名列表）
2. 统计所有边界框面积（归一化后的 w*h）分布，并画直方图（附带类名列表）
3. 统计各类别的边界框数量，并画条形图（横轴为类别名称，纵轴用 log 尺度）
"""

import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import yaml

# ================== 配置区域 ==================
DATASET_ROOT = r"./blind_merged.v2"  # 改成你的数据集根目录
SPLITS = ["train", "valid", "test"]  # 需要统计的 split
TOP_K_CLASS_LABEL = 10  # 在图里展示前 K 个类别名（避免太挤）
# ===========================================


def load_class_names(dataset_root: str):
    yaml_path = os.path.join(dataset_root, "data.yaml")
    if not os.path.exists(yaml_path):
        print(f"[WARN] data.yaml not found at: {yaml_path}")
        return None

    with open(yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    names = data_cfg.get("names", None)
    if isinstance(names, dict):
        # 可能是 {0: 'person', 1: 'car', ...}
        names = [names[k] for k in sorted(names.keys())]
    return names


def collect_statistics(dataset_root: str):
    """
    遍历所有 labels，收集：
    - 每张图片的框数量 num_boxes_per_image
    - 每个框的面积 areas（用归一化后的 w*h）
    - 每个类别的框数量 class_counter
    """
    num_boxes_per_image = []
    areas = []
    class_counter = Counter()

    total_files = 0
    total_boxes = 0

    for split in SPLITS:
        label_dir = os.path.join(dataset_root, split, "labels")
        if not os.path.exists(label_dir):
            print(f"[INFO] Label directory not found for split '{split}': {label_dir}")
            continue

        for fname in os.listdir(label_dir):
            if not fname.endswith(".txt"):
                continue

            fpath = os.path.join(label_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]

            total_files += 1
            num_boxes = len(lines)
            num_boxes_per_image.append(num_boxes)

            for line in lines:
                parts = line.split()
                if len(parts) < 5:
                    continue
                class_id = int(parts[0])
                w = float(parts[3])
                h = float(parts[4])
                area = w * h

                areas.append(area)
                class_counter[class_id] += 1
                total_boxes += 1

    print(f"[INFO] 共统计到 {total_files} 个标签文件")
    print(f"[INFO] 共统计到 {total_boxes} 个边界框")

    return np.array(num_boxes_per_image), np.array(areas), class_counter


def _add_class_names_box(class_names, title_suffix: str = ""):
    """
    在当前图右侧加一个文本框，展示部分类名。
    """
    if not class_names:
        return

    # 只显示前 K 个，避免太挤
    top_names = class_names[:TOP_K_CLASS_LABEL]
    text = "Classes (top {}):\n".format(len(top_names)) + "\n".join(top_names)
    if len(class_names) > TOP_K_CLASS_LABEL:
        text += "\n..."

    # 在图右边画一个文本框
    plt.gcf().text(
        1.02,
        0.5,
        text,
        fontsize=8,
        va="center",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )

    if title_suffix:
        plt.title(title_suffix)


def plot_num_boxes_hist(num_boxes_per_image: np.ndarray, class_names):
    """
    画每张图片的标注数量分布（横轴：一张图里有多少个框）
    """
    if num_boxes_per_image.size == 0:
        print("[WARN] 没有统计到任何标注，无法绘制图片标注数量分布图。")
        return

    plt.figure(figsize=(9, 5))
    max_boxes = int(num_boxes_per_image.max())
    bins = min(max_boxes, 30)
    bins = max(bins, 10)

    plt.hist(num_boxes_per_image, bins=bins, edgecolor="black")
    plt.xlabel("Objects per Image")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    _add_class_names_box(class_names, "Number of Objects per Image")
    plt.tight_layout(rect=[0, 0, 0.8, 1])  # 给右侧文本留点空间
    plt.show()


def plot_area_hist(areas: np.ndarray, class_names):
    """
    画边界框面积分布（归一化 w*h）
    """
    if areas.size == 0:
        print("[WARN] 没有统计到任何边界框，无法绘制面积分布图。")
        return

    plt.figure(figsize=(9, 5))

    plt.hist(areas, bins=50, edgecolor="black")
    plt.xlabel("Area (normalized)")
    plt.ylabel("Count")
    plt.yscale("log")  # y 轴取对数，解决长尾 & 极不均衡
    plt.grid(alpha=0.3)
    _add_class_names_box(class_names, "Bounding Box Area Distribution (log scale)")
    plt.tight_layout(rect=[0, 0, 0.8, 1])
    plt.show()


def plot_class_freq(class_counter: Counter, class_names):
    """
    画各类别的框数量分布（横轴为类别名称，纵轴用 log 尺度来缓解不均衡）
    """
    if not class_counter:
        print("[WARN] 没有统计到任何类别计数，无法绘制类别频次图。")
        return

    # 按数量从大到小排序
    items = sorted(class_counter.items(), key=lambda x: x[1], reverse=True)
    ids = [item[0] for item in items]
    counts = np.array([item[1] for item in items], dtype=float)
    total = counts.sum()
    names = [class_names[i] if class_names and i < len(class_names) else str(i) for i in ids]

    plt.figure(figsize=(max(10, len(names) * 0.6), 6))
    x = np.arange(len(names))
    plt.bar(x, counts, edgecolor="black")

    plt.xticks(x, names, rotation=45, ha="right")
    plt.xlabel("Class")
    plt.ylabel("Number of Objects (log scale)")
    plt.yscale("log")  # **关键：纵轴用 log 尺度，解决类别极度不均衡**
    plt.grid(axis="y", alpha=0.3)

    # 在图上加一个小文本，显示总框数 & 最大/最小类别占比
    max_count = counts.max()
    min_count = counts.min()
    info_text = (
        f"Total boxes: {int(total)}\n"
        f"Max class: {int(max_count)} ({max_count/total:.1%})\n"
    )
    plt.gcf().text(
        0.01,
        0.95,
        info_text,
        fontsize=8,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )

    plt.title("Object Count per Class (log scale)")
    plt.tight_layout()
    plt.show()


def main():
    print(f"[INFO] 数据集根目录: {DATASET_ROOT}")
    class_names = load_class_names(DATASET_ROOT)
    if class_names is not None:
        print(f"[INFO] 类别数: {len(class_names)}")
        print("[INFO] 部分类别:", class_names[:10], "..." if len(class_names) > 10 else "")

    num_boxes_per_image, areas, class_counter = collect_statistics(DATASET_ROOT)

    print("[INFO] 开始绘制每张图片的标注数量分布图...")
    plot_num_boxes_hist(num_boxes_per_image, class_names)

    print("[INFO] 开始绘制边界框面积分布图...")
    plot_area_hist(areas, class_names)

    print("[INFO] 开始绘制各类别频次分布图（log 尺度）...")
    plot_class_freq(class_counter, class_names)


if __name__ == "__main__":
    main()
