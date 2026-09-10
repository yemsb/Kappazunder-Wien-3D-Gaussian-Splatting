---
title: Masking for Dynamic Objects
weight: 40
bookToC: true
---

# Masking for Dynamic Objects

Segmentation masks exclude moving objects (vehicles, pedestrians) from 3DGS training to prevent floaters, ghosting, and convergence issues.

## Overview

Kappazunder captures moving traffic and pedestrians. These violate the static scene assumption of 3DGS and cause:
- **Floaters** — spurious points that don't correspond to real surfaces  
- **Ghosting** — semi-transparent artifacts where objects moved between frames  
- **Convergence failures** — training instability in highly dynamic areas  

Binary masks tell the 3DGS trainer to ignore pixels within masked regions, treating them as "unknown" rather than forcing the model to explain moving objects.

---

## 1. Masking Strategies

### Capture Vehicle Mask (Geometric)

The Kappazunder vehicle appears in every image (cameras are mounted on it). This static mask is geometric and predictable:
- Project the vehicle's 3D bounding box into each camera frame using known mounting positions, intrinsics, and approximate vehicle dimensions  
- Generate a binary mask (255 = masked, 0 = use for training)  
- Typically occupies the lower portion of front/back/side-facing cameras  

### Dynamic Object Masks (Semantic Segmentation)

Moving objects like other cars, pedestrians, and cyclists require semantic segmentation:
- **YOLO v8/v11** for real-time object detection and segmentation  
- **SAM (Segment Anything Model)** for high-quality mask refinement  
- **Manual editing** via Jupyter notebooks for complex cases  

---

## 2. Mask Generation Workflow

### Option A: YOLO Segmentation (Recommended)

The `prepare_yolo_database.py` script automates YOLO-based mask generation:

#### Prerequisites
```bash
pip install ultralytics opencv-python tqdm
```

#### Usage
```bash
cd yolo_segmentation
python prepare_yolo_database.py \
    --images ../datasets/<scene_name>/images \
    --output ./yolo_finetune_images \
    --model yolo11n-seg.pt
```

#### Process
1. **Object Detection** — YOLO identifies objects of interest (car, person, bicycle, etc.)  
2. **Segmentation** — YOLO outputs polygon masks for each detected object  
3. **Format Conversion** — polygons → YOLO segmentation format (normalized coordinates)  
4. **Train/Val Split** — 80/20 split for potential future training  
5. **data.yaml Creation** — defines classes and paths  

#### Output Structure
```
yolo_finetune_images/
├── data.yaml           # YOLO configuration
├── images/
│   ├─ train/           # 80% of images
│   └─ val/             # 20% of images
└── labels/
    ├─ train/
    │  └─ frame_000001.txt  # YOLO format: class x_center y_center width height [x1 y1 x2 y2 ...]
    └─ val/
       └─ frame_000001.txt
```

#### data.yaml Example
```yaml
train: ../yolo_finetune_images/images/train
val: ../yolo_finetune_images/images/val

nc: 2  # number of classes
names: ['car', 'person']
```

### Option B: SAM-Assisted Masking (Highest Quality)

For critical applications or complex scenes:
1. Run YOLO to get initial detections and bounding boxes  
2. Use SAM to refine masks within those boxes (produces tighter boundaries)  
3. Manually correct any remaining errors  
4. Save as binary mask images  

The newer pipeline uses Spirula's automatic mask generation based on SAM3, which detects cars and people automatically.

### Option C: Manual Masking (Small Areas)

Use `mask_car_parts.ipynb` to:
1. Load images sequentially  
2. Draw polygons over moving objects  
3. Save as binary mask images (same naming as input images)  
4. Export to YOLO format if desired  

---

## 3. Mask Formats

### Binary Mask Images (for 3DGS use)

- **Format**: Same dimensions as input images, 8-bit grayscale or binary PNG  
- **Encoding**:  
  - White (255) = **masked** (ignore this pixel during training)  
  - Black (0) = **unmasked** (use this pixel for training)  
- **Naming**: Must match image filename: `frame_000001.jpg` ↔ `frame_000001.png`  
- **Placement**: In a `masks/` subdirectory parallel to `images/`  

### YOLO Segmentation Format (for training/detection)

Each line in `.txt` file:
```
<class_id> <x_center> <y_center> <width> <height> <x1> <y1> <x2> <y2> ... <xn> <yn>
```
Where:
- `<class_id>`: Integer class index (0-based, matches `data.yaml` names)  
- `<x_center> <y_center> <width> <height>`: Bounding box in normalized coordinates [0,1]  
- `<x1> <y1> ... <xn> <yn>`: Polygon vertices in normalized coordinates [0,1], ≥3 points  

Example for a person (class 1) at image center:
```
1 0.5 0.5 0.3 0.6 0.4 0.2 0.6 0.2 0.6 0.8 0.4 0.8
```

---

## 4. Using Masks in 3DGS Training

### Spirula Studio

1. **Import COLMAP project** as usual  
2. **Enable masking** in the import/settings dialog  
3. **Specify mask directory** — point to `datasets/<scene_name>/masks/`  
4. **Choose mask format** — binary images (white = masked)  
5. **Optional: mask dilation** — expand masks by 2–5 pixels to catch edges  
6. **Start training** — masked pixels are skipped during loss computation  

Spirula can also generate masks automatically using SAM3 for cars and people.

### Lichtfeld Studio

1. **Create new project** from COLMAP export  
2. **Navigate to Masking tab** in project settings  
3. **Add mask source** — select directory containing mask images  
4. **Set mask interpretation** — "white = masked"  
5. **Optional: mask dilation** — expand masks slightly  
6. **Start training** — Lichtfeld skips masked pixels in rendering loss  

---

## 5. Best Practices

### When to Mask

- ✅ **Always** mask the capture vehicle (appears in every image)  
- ✅ **Recommended** mask moving vehicles, pedestrians, cyclists in busy streets  
- ⚠️ **Optional** mask frequently changing elements (flags, construction, signage)  
- ❌ **Avoid over-masking** — don't mask static scene elements  

### Mask Quality Guidelines

- **Accuracy over completeness** — better to miss some pixels than include static ones  
- **Slight over-masking is safer** — ignoring a few good pixels is better than fitting moving ones  
- **Account for motion blur** — for fast-moving objects, mask a slightly larger region (2–5 pixel dilation)  
- **Check temporal consistency** — the same object should be masked similarly across frames  

### Performance Impact

Masking has minimal performance impact:
- **Loading time** — negligible overhead (masks loaded per-batch)  
- **Memory usage** — same size as images, loaded on-demand  
- **Training speed** — actually *faster* since masked pixels are skipped in loss computation  

---

## 6. Configuration Integration

Masks are automatically discovered if present. In `build_colmap_selection.py`:
1. Script checks for mask files at `<data_path>/Bild-Rohdaten/Trajektorie_*/Sensor_*/masks/<image_name>.png`  
2. If found, masks are copied to `datasets/<scene_name>/masks/` during image staging  
3. The `mask_path` column in the metadata tracks which images have masks  

No explicit configuration required — place masks in the expected location and they will be used.

---

## 7. Example: Stadtpark Masking

For the Stadtpark scene:

```bash
# 1. Generate masks with YOLO
cd yolo_segmentation
python prepare_yolo_database.py \
    --images ../datasets/stadtpark/images \
    --output ./yolo_stadtpark_masks \
    --model yolo11n-seg.pt \
    --classes 0 1 2  # car, bicycle, person

# 2. Convert YOLO polygons to binary masks (if not already done)
python yolo_to_binary_masks.py \
    --labels ./yolo_stadtpark_masks/labels/train \
    --images ../datasets/stadtpark/images \
    --output ../datasets/stadtpark/masks \
    --threshold 0.5

# 3. Verify a sample
# Check that masked regions show white pixels over moving objects
```

Alternatively, use Spirula's automatic SAM3-based mask generation during import:
1. Import COLMAP project into Spirula Studio  
2. Enable **"Generate masks automatically"**  
3. Select classes: **Cars** and **People**  
4. Spirula generates masks on-the-fly during initialization  

---

## 8. Troubleshooting

### Masks Not Appearing in Trainer

- **Check format** — ensure binary masks use 255=masked, 0=unmasked (not inverted)  
- **Verify paths** — trainer must be pointed to correct mask directory  
- **Check naming** — mask files must match image filenames exactly (extension may differ: `.jpg` → `.png`)  
- **Confirm loading** — check trainer settings for mask overlay preview  

### Training Still Shows Moving Objects

- **Mask too small** — expand masks with 2–5 pixel dilation  
- **Misalignment** — verify masks align with objects in images (check a few samples visually)  
- **Wrong class** — ensure you're masking the right objects (e.g., masking cars but missing buses/trucks)  
- **Threshold issues** — for probability masks, ensure binarization threshold is appropriate  

### Over-Masking (Loss of Detail)

- **Too much dilation** — reduce mask expansion  
- **Static objects masked** — review YOLO class selection, exclude parked cars if they're truly static  
- **Manual correction needed** — use Jupyter notebook to edit masks for specific frames  

---

## 9. Advanced Techniques

### Temporal Consistency

For objects appearing across multiple frames:
1. Generate mask for first appearance  
2. Propagate to subsequent frames using optical flow  
3. Manually verify and correct drift  

### Instance-Aware Masking

Different masks for different instances (e.g., keep parked cars, remove moving ones):
1. Use YOLO instance segmentation  
2. Manually select which instances to mask  
3. Generate separate mask sets per instance ID  

### Semi-Automatic Refinement

1. Train initial 3DGS model without masks  
2. Render novel views and compare to input images (compute per-pixel loss)  
3. Identify regions with high reprojection error (likely moving objects)  
4. Generate masks for those regions  
5. Retrain with masks  

---

## 10. Next Steps

- [Training Guide](../../training/overview) — loading the export into Spirula or Lichtfeld Studio  
- [Troubleshooting](../../troubleshooting/common-issues) — fixing masking-specific problems  
