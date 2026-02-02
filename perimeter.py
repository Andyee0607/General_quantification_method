"""
Universal Organelle Perimeter Quantifier
---------------------------------------
通用细胞器实例周长统计 + 轮廓可视化 + CSV导出

支持:
✓ RGB instance mask (SAM / micro-sam)
✓ label mask
✓ binary mask (自动连通域)

输出:
✓ perimeter overlay image
✓ csv table
✓ statistics txt

Author: ChatGPT
Version: 1.0 universal
"""

import os
import cv2
import numpy as np
import pandas as pd
from typing import Dict, Optional
from skimage import measure


# =====================================================
# =============== mask 转 instance ====================
# =====================================================

def rgb_to_instance(mask):
    return (
        mask[:, :, 0].astype(np.int32) * 256 * 256 +
        mask[:, :, 1].astype(np.int32) * 256 +
        mask[:, :, 2].astype(np.int32)
    )


def binary_to_instances(binary):
    _, cc = cv2.connectedComponents(binary.astype(np.uint8))
    return cc


def ensure_instance_mask(mask):
    if mask.ndim == 3:
        return rgb_to_instance(mask)

    uniq = np.unique(mask)

    if len(uniq) <= 2:
        return binary_to_instances(mask)

    return mask.astype(np.int32)


# =====================================================
# =============== 周长计算核心 =========================
# =====================================================

def compute_perimeters(instance_img):
    """
    使用 regionprops 计算更准确的 perimeter
    """
    props = measure.regionprops(instance_img)

    result = {}

    for p in props:
        uid = p.label
        perim = p.perimeter
        result[uid] = float(perim)

    return result


# =====================================================
# =============== 可视化轮廓 ===========================
# =====================================================

def create_overlay(original, instance_img, color=(255, 0, 0)):
    """
    生成轮廓叠加图
    """
    if original.ndim == 2:
        overlay = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    else:
        overlay = original.copy()

    ids = np.unique(instance_img)
    ids = ids[ids != 0]

    for uid in ids:
        mask = (instance_img == uid).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        cv2.drawContours(overlay, contours, -1, color, 1)

    return overlay


# =====================================================
# =============== 主函数 ===============================
# =====================================================

def run_organelle_perimeter(
        image_path: str,
        mask: np.ndarray,
        out_dir: str,
        organelle_name: str = "organelle",
        pixel_size_um: Optional[float] = None
) -> Dict[str, str]:
    """
    Napari / 脚本统一接口
    """

    os.makedirs(out_dir, exist_ok=True)

    original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if original is None:
        raise RuntimeError(f"Cannot load image: {image_path}")

    instance_img = ensure_instance_mask(mask)

    # -----------------------------
    # 计算 perimeter
    # -----------------------------
    perims = compute_perimeters(instance_img)

    if len(perims) == 0:
        raise RuntimeError("No instances found")

    # -----------------------------
    # 保存 CSV
    # -----------------------------
    df = pd.DataFrame({
        "instance_id": list(perims.keys()),
        "perimeter_pixels": list(perims.values())
    })

    if pixel_size_um:
        df["perimeter_um"] = df["perimeter_pixels"] * pixel_size_um

    csv_path = os.path.join(out_dir, f"{organelle_name}_perimeter_table.csv")
    df.to_csv(csv_path, index=False)

    # -----------------------------
    # 统计信息
    # -----------------------------
    txt_path = os.path.join(out_dir, f"{organelle_name}_perimeter_statistics.txt")

    with open(txt_path, "w") as f:
        f.write(f"{organelle_name} perimeter statistics\n")
        f.write("=" * 40 + "\n")
        f.write(f"count: {len(df)}\n")
        f.write(f"min: {df.perimeter_pixels.min():.2f}\n")
        f.write(f"max: {df.perimeter_pixels.max():.2f}\n")
        f.write(f"mean: {df.perimeter_pixels.mean():.2f}\n")
        f.write(f"median: {df.perimeter_pixels.median():.2f}\n")

    # -----------------------------
    # overlay 图像
    # -----------------------------
    overlay = create_overlay(original, instance_img)

    base = os.path.splitext(os.path.basename(image_path))[0]
    overlay_path = os.path.join(out_dir, f"{base}_{organelle_name}_perimeter_overlay.png")

    cv2.imwrite(overlay_path, overlay)

    print(f"✓ saved → {overlay_path}")
    print(f"✓ saved → {csv_path}")
    print(f"✓ saved → {txt_path}")

    return {
        "overlay": overlay_path,
        "csv": csv_path,
        "stats": txt_path
    }


# =====================================================
# =============== 示例 ================================
# =====================================================

if __name__ == "__main__":
    img = "test.png"
    mask = cv2.imread("label.png", cv2.IMREAD_UNCHANGED)

    run_organelle_perimeter(
        img,
        mask,
        "./output",
        organelle_name="mitochondria",
        pixel_size_um=0.1
    )