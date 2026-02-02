"""
Universal Organelle Area Quantifier
-----------------------------------
通用细胞器面积统计 + 颜色映射 + 统计表生成

支持:
✓ RGB 实例分割 mask
✓ label mask
✓ 二值语义 mask（自动连通域实例化）

Author: ChatGPT
Version: 4.0 (universal)
"""

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Optional


# =====================================================
# =============== 工具函数 =============================
# =====================================================

def rgb_to_instance_id(label: np.ndarray) -> np.ndarray:
    """RGB → 唯一实例ID"""
    return (
        label[:, :, 0].astype(np.int32) * 256 * 256 +
        label[:, :, 1].astype(np.int32) * 256 +
        label[:, :, 2].astype(np.int32)
    )


def label_to_instances(binary_or_label: np.ndarray) -> np.ndarray:
    """
    对语义mask做 connected components
    """
    _, cc = cv2.connectedComponents(binary_or_label.astype(np.uint8))
    return cc


def create_gradient_colorbar(height, width, colors):
    bar = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(height):
        r = i / (height - 1)
        pos = r * (len(colors) - 1)

        idx = int(pos)
        idx = min(idx, len(colors) - 2)

        t = pos - idx

        c1 = np.array(colors[idx])
        c2 = np.array(colors[idx + 1])

        bar[i] = (c1 * (1 - t) + c2 * t).astype(np.uint8)

    return bar


# =====================================================
# =============== 主函数 ===============================
# =====================================================

def run_organelle_area(
        mask: np.ndarray,
        out_dir: str,
        organelle_name: str = "organelle",
        pixel_size_um: Optional[float] = None,
        bins_num: int = 7
) -> Dict[str, str]:
    """
    通用面积计算函数

    Parameters
    ----------
    mask:
        RGB / label / binary mask
    pixel_size_um:
        像素尺寸 (μm)，提供则自动计算 μm²
    """

    os.makedirs(out_dir, exist_ok=True)

    # -------------------------------------------------
    # 1️⃣ mask 统一为 instance id
    # -------------------------------------------------
    if mask.ndim == 3:
        instance_img = rgb_to_instance_id(mask)
    else:
        if np.unique(mask).size <= 2:
            instance_img = label_to_instances(mask)
        else:
            instance_img = mask.astype(np.int32)

    h, w = instance_img.shape

    # -------------------------------------------------
    # 2️⃣ 统计面积
    # -------------------------------------------------
    ids, counts = np.unique(instance_img, return_counts=True)

    areas = {}
    for uid, c in zip(ids, counts):
        if uid == 0:
            continue
        areas[uid] = int(c)

    if len(areas) == 0:
        raise RuntimeError("No instances found in mask")

    area_arr = np.array(list(areas.values()))

    min_area = area_arr.min()
    max_area = area_arr.max()

    # -------------------------------------------------
    # 3️⃣ 分箱
    # -------------------------------------------------
    bins = np.linspace(min_area, max_area, bins_num + 1)

    colors = [
        (255, 0, 0),
        (255, 128, 0),
        (255, 255, 0),
        (0, 255, 0),
        (0, 255, 255),
        (0, 0, 255),
        (128, 0, 255),
    ][:bins_num]

    output = np.zeros((h, w, 3), dtype=np.uint8)

    for uid, area in areas.items():
        idx = np.digitize(area, bins) - 1
        idx = np.clip(idx, 0, bins_num - 1)
        output[instance_img == uid] = colors[idx]

    # -------------------------------------------------
    # 4️⃣ 生成 colorbar 画布
    # -------------------------------------------------
    border = 30
    gap = 25
    bar_w = 60

    total_w = border + w + gap + bar_w + 150 + border
    total_h = border + h + border

    canvas = np.ones((total_h, total_w, 3), dtype=np.uint8) * 255
    canvas[border:border+h, border:border+w] = output

    bar = create_gradient_colorbar(h - 60, bar_w, colors)

    bar_top = border + 30
    bar_left = border + w + gap

    canvas[bar_top:bar_top+bar.shape[0], bar_left:bar_left+bar_w] = bar

    # -------------------------------------------------
    # 5️⃣ 添加文字
    # -------------------------------------------------
    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except:
        font = ImageFont.load_default()

    draw.text((bar_left, bar_top - 25),
              f"{organelle_name} Area (pixels)",
              fill=(0, 0, 0),
              font=font)

    draw.text((bar_left + bar_w + 10, bar_top),
              f"max: {max_area}",
              fill=(0, 0, 0),
              font=font)

    draw.text((bar_left + bar_w + 10, bar_top + bar.shape[0] - 20),
              f"min: {min_area}",
              fill=(0, 0, 0),
              font=font)

    final_img = np.array(img)

    # -------------------------------------------------
    # 6️⃣ 保存图像
    # -------------------------------------------------
    img_path = os.path.join(out_dir, f"{organelle_name}_area_colormap.png")
    Image.fromarray(final_img).save(img_path)

    # -------------------------------------------------
    # 7️⃣ CSV + 统计文件
    # -------------------------------------------------
    df = pd.DataFrame({
        "instance_id": list(areas.keys()),
        "area_pixels": list(areas.values())
    })

    if pixel_size_um:
        df["area_um2"] = df["area_pixels"] * (pixel_size_um ** 2)

    csv_path = os.path.join(out_dir, f"{organelle_name}_area_table.csv")
    df.to_csv(csv_path, index=False)

    txt_path = os.path.join(out_dir, f"{organelle_name}_area_statistics.txt")

    with open(txt_path, "w") as f:
        f.write(f"{organelle_name} statistics\n")
        f.write("="*40 + "\n")
        f.write(f"count: {len(df)}\n")
        f.write(f"min: {min_area}\n")
        f.write(f"max: {max_area}\n")
        f.write(f"mean: {df.area_pixels.mean():.2f}\n")
        f.write(f"median: {df.area_pixels.median():.2f}\n")

    print(f"✓ saved → {img_path}")
    print(f"✓ saved → {csv_path}")
    print(f"✓ saved → {txt_path}")

    return {
        "colormap": img_path,
        "csv": csv_path,
        "stats": txt_path
    }


# =====================================================
# =============== 示例运行 =============================
# =====================================================

if __name__ == "__main__":
    label_path = r"D:\PycharmProjects\UnetSeg\quantify_data\mito\label\m01.png"
    out_dir = r"D:\PycharmProjects\UnetSeg\quantify_data\mito\area"

    label = cv2.imread(label_path)
    label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)

    run_organelle_area(
        mask=label,
        out_dir=out_dir,
        organelle_name="mitochondria",
        pixel_size_um=0.1
    )