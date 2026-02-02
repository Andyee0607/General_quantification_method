import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skimage.morphology import skeletonize, binary_dilation
from skimage import measure


# ======================================================
# mask → instance (统一格式转换)
# ======================================================

def rgb_to_instance(mask):
    return (
        mask[:, :, 0].astype(np.int32) * 256 * 256 +
        mask[:, :, 1].astype(np.int32) * 256 +
        mask[:, :, 2].astype(np.int32)
    )


def binary_to_instance(binary):
    _, cc = cv2.connectedComponents(binary.astype(np.uint8))
    return cc


def ensure_instance_mask(mask):
    if mask.ndim == 3:
        return rgb_to_instance(mask)

    uniq = np.unique(mask)
    if len(uniq) <= 2:
        return binary_to_instance(mask)

    return mask.astype(np.int32)


# ======================================================
# skeleton 核心计算
# ======================================================

def skeleton_metrics(skel):
    """
    计算：
    - length
    - branch points
    - end points
    """

    length = np.sum(skel)

    kernel = np.array([
        [1,1,1],
        [1,10,1],
        [1,1,1]
    ])

    conv = cv2.filter2D(skel.astype(np.uint8), -1, kernel)

    neighbors = conv - 10

    end_points = np.sum((skel == 1) & (neighbors == 1))
    branch_points = np.sum((skel == 1) & (neighbors >= 3))

    return length, branch_points, end_points


# ======================================================
# 主 skeleton 处理
# ======================================================

def compute_skeleton(instance_img):

    unique_ids = np.unique(instance_img)
    unique_ids = unique_ids[unique_ids != 0]

    skel_map = np.zeros_like(instance_img, dtype=np.uint8)

    records = []

    for uid in unique_ids:
        obj = instance_img == uid

        if obj.sum() < 5:
            continue

        skel = skeletonize(obj)
        skel = skel.astype(np.uint8)

        length, branches, ends = skeleton_metrics(skel)

        skel_map[skel > 0] = 255

        records.append((uid, length, branches, ends))

    return skel_map, records


# ======================================================
# 可视化
# ======================================================

def overlay_skeleton(gray, skel_map):

    color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    mask = binary_dilation(skel_map > 0)

    color[mask] = (0, 0, 255)

    return color


def create_visual(original, label, skel_map, overlay, save_path):

    plt.figure(figsize=(18, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(original, cmap='gray')
    plt.title("Original")
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(label)
    plt.title("Instance mask")
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(skel_map, cmap='gray')
    plt.title("Skeleton")
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title("Overlay")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ======================================================
# ⭐ 主接口（统一科研 pipeline）
# ======================================================

def run_organelle_skeleton(image_path, mask, out_dir, organelle_name="organelle"):

    os.makedirs(out_dir, exist_ok=True)

    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    inst = ensure_instance_mask(mask)

    skel_map, records = compute_skeleton(inst)

    df = pd.DataFrame(
        records,
        columns=["instance_id", "length_px", "branch_points", "end_points"]
    )

    csv_path = os.path.join(out_dir, f"{organelle_name}_skeleton_metrics.csv")
    df.to_csv(csv_path, index=False)

    skel_path = os.path.join(out_dir, f"{organelle_name}_skeleton.png")
    cv2.imwrite(skel_path, skel_map)

    overlay = overlay_skeleton(gray, skel_map)

    overlay_path = os.path.join(out_dir, f"{organelle_name}_skeleton_overlay.png")
    cv2.imwrite(overlay_path, overlay)

    vis_path = os.path.join(out_dir, f"{organelle_name}_skeleton_visual.png")
    create_visual(gray, inst, skel_map, overlay, vis_path)

    print(f"[{organelle_name}] skeleton finished ✓  (n={len(df)})")

    return {
        "csv": csv_path,
        "skeleton": skel_path,
        "overlay": overlay_path,
        "visual": vis_path
    }