---
title: Training Overview
weight: 10
bookToC: true
---

# Training Overview

Loading your pipeline output into Spirula Studio or Lichtfeld Studio for 3D Gaussian Splatting training.

## Overview

After completing the pipeline steps (`build_colmap_selection.py` and `laz_to_ply.py`), you'll have a COLMAP-format export ready for 3DGS training:

```
datasets/<scene_name>/
├── depths/					# Depth maps (optional)
│   └─ frame_*.png          
├── images/					# Selected images
│   └─ frame_*.jpg          
├── masks/                  # Ignore masks (optional)
│   └─ frame_*.png
├── normals/				# Normal maps (optional)
│   └─ frame_*.png
├── sparse/0/
│   ├─ cameras.txt          # Camera intrinsics
│   ├─ images.txt           # Camera poses (quaternions + translations)
│   ├─ points3D.txt         # Initial point cloud (XYZ + RGB)
│   └─ points3D.ply         # Initial point cloud from LiDAR
├── scene_origin.txt        # Offset applied to center the scene
└── selection_preview.png	# Preview of selected images
```

Both Spirula Studio and Lichtfeld Studio accept this standard COLMAP export format.

---

## Training Best Practices

### Initial Point Cloud Density

Adapt voxel size so **10–50% of the final Gaussian count** is initialized from the LiDAR point cloud:
- If final model has 2M Gaussians, initialize with 200K–1M points  
- Too few initial points → slow convergence, many iterations spent growing  
- Too many initial points → slow initialization, excessive pruning early on  

### Iteration Count

Start with a fast preview to verify alignment:
- **7K iterations** (5–10 min) → check for gross alignment issues  
- If preview looks good, train to 30K (20–40 min) for production quality  
- Only go beyond 50K if you need highest quality and have time  

### Masking Strategy

- **Always** mask the capture vehicle (appears in every image with front-facing camera)  
- **Recommended** mask moving vehicles and pedestrians in busy streets  
- Mask frequently changing elements (flags, construction)  

---

## Next Steps

- [Stadtpark Walkthrough](./stadtpark) — see how these best practices were applied to a real-world reconstruction
- [Kolonitzpark Walkthrough](./kolonitzpark) — another example of applying these best practices to a different scene
