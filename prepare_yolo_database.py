"""
Prepare a YOLO-seg dataset from a folder of images + black/white masks.

Converts each mask into normalized polygon labels (YOLO segmentation format),
splits images into train/val, copies everything into the expected folder
structure, and writes a data.yaml.

Resulting layout:

    ./yolo_finetune_images/
        images/train/*.jpg
        images/val/*.jpg
        labels/train/*.txt
        labels/val/*.txt
        data.yaml

Label format (one .txt per image, same basename as the image):
    <class_id> x1 y1 x2 y2 ... xn yn   (all coords normalized 0-1)
    one line per separate blob/instance found in the mask
"""

import os
import shutil
import random
import cv2
import numpy as np

# ----------------------------- CONFIG ---------------------------------------

IMAGES_PATH = "../LiDAR_kappazunder_stadtpark/Los_6A/Bild-Rohdaten/Trajektorie_15767/Sensor_110021/"
MASKS_PATH = IMAGES_PATH + "masks/"

OUTPUT_ROOT = "./yolo_finetune_images/"

CLASS_NAMES = ["object"]  # edit if you have more than one class

TRAIN_RATIO = 0.85
RANDOM_SEED = 42

MASK_BINARY_THRESHOLD = 127       # pixel value above this counts as foreground
MIN_CONTOUR_AREA_FRAC = 0.0005    # skip contours smaller than this fraction of image area
                                   # (0.0005 = 0.05% -> ~2540px^2 on a 7130x7130 image)
POLY_APPROX_EPS_FRAC = 0.0001      # contour simplification factor (relative to perimeter)

# -----------------------------------------------------------------------------


def find_pairs(images_path, masks_path):
    """
    Match images to masks by sorted alphabetical order (filenames don't correspond
    directly, but both folders have a 1:1 correspondence when sorted).
    """
    image_files = sorted(f for f in os.listdir(images_path) if f.lower().endswith(".jpg"))
    mask_files = sorted(f for f in os.listdir(masks_path) if f.lower().endswith(".jpg"))

    if len(image_files) != len(mask_files):
        print(f"[WARN] Mismatch: {len(image_files)} images vs {len(mask_files)} masks. "
              f"Truncating to the shorter list -- check this is correct!")

    n = min(len(image_files), len(mask_files))
    pairs = list(zip(image_files[:n], mask_files[:n]))

    print("First few matched pairs (sanity check):")
    for img_name, mask_name in pairs[:3]:
        print(f"       {img_name}  <->  {mask_name}")
    print(f"       ... {n} pairs total\n")

    return pairs


def mask_to_yolo_lines(mask_path, class_id=0):
    """Convert a B/W mask into YOLO-seg polygon lines (normalized)."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")

    h, w = mask.shape[:2]
    min_area = MIN_CONTOUR_AREA_FRAC * h * w

    _, binary = cv2.threshold(mask, MASK_BINARY_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lines = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        peri = cv2.arcLength(cnt, True)
        eps = POLY_APPROX_EPS_FRAC * peri
        approx = cv2.approxPolyDP(cnt, eps, True)

        if len(approx) < 3:
            continue  # not a valid polygon

        coords = []
        for point in approx.reshape(-1, 2):
            x_norm = point[0] / w
            y_norm = point[1] / h
            coords.append(f"{x_norm:.6f}")
            coords.append(f"{y_norm:.6f}")

        lines.append(f"{class_id} " + " ".join(coords))

    return lines


def draw_sanity_check(output_root):
    """
    Pick one random image from the processed dataset, draw its polygon labels
    on top, and save it next to data.yaml so you can visually verify the
    mask -> contour -> polygon conversion worked correctly.
    """
    candidates = []
    for split in ("train", "val"):
        img_dir = os.path.join(output_root, "images", split)
        lbl_dir = os.path.join(output_root, "labels", split)
        for fname in os.listdir(img_dir):
            candidates.append((split, fname))

    if not candidates:
        print("[WARN] No processed images found, skipping sanity check image.")
        return

    split, fname = random.choice(candidates)
    img_path = os.path.join(output_root, "images", split, fname)
    label_path = os.path.join(output_root, "labels", split, os.path.splitext(fname)[0] + ".txt")

    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARN] Could not read {img_path} for sanity check.")
        return
    h, w = img.shape[:2]

    with open(label_path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    if not lines:
        print(f"[WARN] '{fname}' ({split}) has no polygons -- picking another for sanity check.")
        candidates.remove((split, fname))
        if candidates:
            split, fname = random.choice(candidates)
            img_path = os.path.join(output_root, "images", split, fname)
            label_path = os.path.join(output_root, "labels", split, os.path.splitext(fname)[0] + ".txt")
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            with open(label_path, "r") as f:
                lines = [l.strip() for l in f if l.strip()]

    for line in lines:
        parts = line.split()
        coords = list(map(float, parts[1:]))
        pts = np.array(
            [[int(coords[i] * w), int(coords[i + 1] * h)] for i in range(0, len(coords), 2)],
            dtype=np.int32,
        )
        cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 255), thickness=8)

    out_path = os.path.join(output_root, "sanity_check.jpg")
    cv2.imwrite(out_path, img)
    print(f"Sanity check image ('{split}/{fname}') written to: {os.path.abspath(out_path)}")


def main():
    random.seed(RANDOM_SEED)

    pairs = find_pairs(IMAGES_PATH, MASKS_PATH)
    if not pairs:
        print("No matching image/mask pairs found. Check your paths/filenames.")
        return

    random.shuffle(pairs)
    split_idx = int(len(pairs) * TRAIN_RATIO)
    train_files = pairs[:split_idx]
    val_files = pairs[split_idx:]

    print(f"Found {len(pairs)} pairs -> {len(train_files)} train / {len(val_files)} val")

    # create folder structure
    for split in ("train", "val"):
        os.makedirs(os.path.join(OUTPUT_ROOT, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_ROOT, "labels", split), exist_ok=True)

    def process_split(file_list, split_name):
        empty_label_count = 0
        for img_name, mask_name in file_list:
            src_img = os.path.join(IMAGES_PATH, img_name)
            src_mask = os.path.join(MASKS_PATH, mask_name)

            dst_img = os.path.join(OUTPUT_ROOT, "images", split_name, img_name)
            shutil.copy2(src_img, dst_img)

            lines = mask_to_yolo_lines(src_mask, class_id=0)
            if not lines:
                empty_label_count += 1

            # label file must share the basename of the IMAGE, not the mask
            label_name = os.path.splitext(img_name)[0] + ".txt"
            dst_label = os.path.join(OUTPUT_ROOT, "labels", split_name, label_name)
            with open(dst_label, "w") as f:
                f.write("\n".join(lines))

        if empty_label_count:
            print(f"[WARN] {empty_label_count} mask(s) in '{split_name}' produced no valid polygon "
                  f"(fully empty mask, or all blobs below MIN_CONTOUR_AREA)")

    process_split(train_files, "train")
    process_split(val_files, "val")

    # write data.yaml
    names_block = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES))
    yaml_content = (
        f"path: {os.path.abspath(OUTPUT_ROOT)}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"\n"
        f"names:\n"
        f"{names_block}\n"
    )
    yaml_path = os.path.join(OUTPUT_ROOT, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"\nDone. Dataset written to: {os.path.abspath(OUTPUT_ROOT)}")
    print(f"data.yaml:\n{yaml_content}")

    draw_sanity_check(OUTPUT_ROOT)


if __name__ == "__main__":
    main()