"""
Universal Organelle Instance Counter
-----------------------------------
通用细胞器实例数量统计 + 可视化 + CSV导出

支持:
✓ RGB instance mask (SAM / micro-sam)
✓ label mask
✓ binary mask (自动 connected components)

Author: ChatGPT
Version: 2.0 universal
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict


# =====================================================
# =============== 工具函数 =============================
# =====================================================

def rgb_to_instance_id(mask):
    """RGB → 唯一实例ID"""
    return (
        mask[:, :, 0].astype(np.int32) * 256 * 256 +
        mask[:, :, 1].astype(np.int32) * 256 +
        mask[:, :, 2].astype(np.int32)
    )


def binary_to_instances(binary):
    """binary mask → connected components"""
    _, cc = cv2.connectedComponents(binary.astype(np.uint8))
    return cc


def ensure_instance_mask(mask):
    """
    自动识别 mask 类型并转换为 instance id
    """
    if mask.ndim == 3:
        return rgb_to_instance_id(mask)

    unique_vals = np.unique(mask)

    # binary mask
    if len(unique_vals) <= 2:
        return binary_to_instances(mask)

    # label mask
    return mask.astype(np.int32)


# =====================================================
# =============== 统计 ================================
# =====================================================

def count_instances(label_img):
    ids, counts = np.unique(label_img, return_counts=True)

    stats = {}
    for uid, c in zip(ids, counts):
        if uid == 0:
            continue
        stats[uid] = int(c)

    return len(stats), stats


# =====================================================
# =============== 可视化 ===============================
# =====================================================

def generate_colors(n):
    """生成视觉可区分颜色"""
    import colorsys

    colors = []
    for i in range(n):
        hue = i / max(n, 1)
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        colors.append(tuple(int(c * 255) for c in rgb))
    return colors


def create_colored_label(label_img):
    h, w = label_img.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)

    ids = np.unique(label_img)
    ids = ids[ids != 0]

    colors = generate_colors(len(ids))

    for i, uid in enumerate(ids):
        out[label_img == uid] = colors[i]

    return out


def visualize(original, label_img, organelle_name, stats, save_path):
    colored = create_colored_label(label_img)

    plt.figure(figsize=(16, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(colored)
    plt.title(f"{organelle_name} Instances")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.bar(range(len(stats)), list(stats.values()))
    plt.xlabel("Instance")
    plt.ylabel("Pixel Area")
    plt.title(f"Count = {len(stats)}")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# =====================================================
# =============== 主接口函数 ===========================
# =====================================================

def run_organelle_count(
        image_path: str,
        mask: np.ndarray,
        out_dir: str,
        organelle_name: str = "organelle"
) -> Dict[str, str]:
    """
    Napari/脚本统一接口

    Parameters
    ----------
    image_path: 原始灰度图
    mask: instance mask
    out_dir: 输出目录
    organelle_name: 细胞器名称
    """

    os.makedirs(out_dir, exist_ok=True)

    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if original is None:
        raise RuntimeError(f"Cannot load image: {image_path}")

    instance_mask = ensure_instance_mask(mask)

    count, stats = count_instances(instance_mask)

    print(f"[{organelle_name}] instance count = {count}")

    # -------------------------------------------------
    # 保存 CSV
    # -------------------------------------------------
    df = pd.DataFrame({
        "instance_id": list(stats.keys()),
        "area_pixels": list(stats.values())
    })

    csv_path = os.path.join(out_dir, f"{organelle_name}_instance_table.csv")
    df.to_csv(csv_path, index=False)

    # -------------------------------------------------
    # 保存可视化
    # -------------------------------------------------
    base = os.path.splitext(os.path.basename(image_path))[0]
    fig_path = os.path.join(out_dir, f"{base}_{organelle_name}_count.png")

    visualize(original, instance_mask, organelle_name, stats, fig_path)

    print(f"✓ saved → {fig_path}")
    print(f"✓ saved → {csv_path}")

    return {
        "visualization": fig_path,
        "csv": csv_path
    }


# =====================================================
# =============== 示例 ================================
# =====================================================

if __name__ == "__main__":
    img_path = "test.png"
    label = cv2.imread("label.png", cv2.IMREAD_UNCHANGED)

    run_organelle_count(
        img_path,
        label,
        "./output",
        organelle_name="mitochondria"
    )