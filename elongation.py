import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import measure


# =====================================================
# mask → instance
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
# 核心计算
# =====================================================

def compute_elongation(instance_img):
    props = measure.regionprops(instance_img)
    results = []

    for r in props:
        area = r.area
        perim = r.perimeter_crofton

        if area == 0 or perim == 0:
            continue

        circ = 4 * np.pi * area / (perim ** 2)
        circ = min(circ, 1.0)
        elong = 1.0 / circ

        results.append((r.label, area, perim, elong))

    return results


# =====================================================
# 可视化
# =====================================================

def create_map(label_img, elong_dict):
    h, w = label_img.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)

    for uid, e in elong_dict.items():
        m = label_img == uid

        if e <= 1.5:
            out[m] = (255, 255, 0)
        elif e <= 3:
            out[m] = (255, 0, 0)
        else:
            out[m] = (0, 0, 255)

    return out


# =====================================================
# 主接口
# =====================================================

def run_organelle_elongation(image_path, mask, out_dir, organelle_name="organelle"):
    os.makedirs(out_dir, exist_ok=True)

    original = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    inst = ensure_instance_mask(mask)

    records = compute_elongation(inst)

    df = pd.DataFrame(
        records,
        columns=["instance_id", "area", "perimeter", "elongation"]
    )

    csv_path = os.path.join(out_dir, f"{organelle_name}_elongation.csv")
    df.to_csv(csv_path, index=False)

    elong_dict = dict(zip(df.instance_id, df.elongation))

    cmap = create_map(inst, elong_dict)
    map_path = os.path.join(out_dir, f"{organelle_name}_elongation_map.png")
    cv2.imwrite(map_path, cv2.cvtColor(cmap, cv2.COLOR_RGB2BGR))

    hist_path = os.path.join(out_dir, f"{organelle_name}_elongation_hist.png")
    plt.hist(df.elongation, bins=30)
    plt.xlabel("Elongation")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(hist_path, dpi=300)
    plt.close()

    print(f"[{organelle_name}] elongation analysis finished ✓")

    return {
        "csv": csv_path,
        "map": map_path,
        "hist": hist_path
    }