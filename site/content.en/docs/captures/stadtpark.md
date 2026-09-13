---
title: Stadtpark Capture Walkthrough
weight: 20
bookToC: true
---

# Stadtpark Capture Walkthrough

This is the first 3DGS reconstruction I attempted with the Kappazunder dataset. 
The [Stadtpark](https://www.wien.gv.at/freizeit/stadtpark) area is a central urban park in Vienna opened in 1862 under Vienna's mayor, Andreas Zelinka (1802-1868). 
The Stadtpark was planned and built under Zelinka's reign, and it features a monument dedicated to the beloved mayor, who [donated his entire salary to the poor](https://www.geschichtewiki.wien.gv.at/Andreas_Zelinka).

## Overview

Stadtpark is an urban park in Vienna featuring:
- Tree-lined paths
- Open grassy areas
- Water features
- Mixed lighting conditions
- Moderate pedestrian traffic

{{< image src="images/frame_000488-ezgif.com-optimize.gif">}}

The immediate area surrounding the Andreas Zelinka monument was chosen from the Kappazunder dataset for this reconstruction. 

{{< hint warning >}}
Medium difficulty for this reconstruction, as the park square is relatively small and not many positions are captured. 
The main challenge is the slight miscalibration of the camera positions, which can be mitigated by running an alignment step.
{{< /hint >}}

### Area of Interest (AoI)
```
# EPSG: 31256
# x_m		y_m
3599.8874	340907.1736
3648.3580	340882.9493
3616.9634	340824.9683
3571.3034	340861.6948
3599.8874	340907.1736
```

{{< image src="images/stadtpark_AoI.png" alt="Stadtpark Area of Interest" caption="Area of Interest (AoI) for the Stadtpark reconstruction." >}}

After [frustum selection](../pipeline/camera-selection/#2-sensor-roles--frustum-geometry) and an acceptance border of 10 meters around the AoI, 350 images (~1.8 GB) were selected.

## First Trials

{{% hint info %}}
As this reconstruction was done before the config workflow was implemented, the steps were performed manually in the notebook [`reduce_stadtpark_data.ipynb`](https://github.com/yemsb/Kappazunder-Wien-3D-Gaussian-Splatting/blob/4490a673224d6b692d71be07f09213e6c3819567/colmap_pipeline/reduce_stadtpark_data.ipynb).
{{% /hint %}}

In initial trials, I used the point cloud and camera poses provided by Kappazunder without any masks. 
I quickly realised that the **alignment is not perfect**, which is especially noticeable in medium- to small-sized objects. 
Furthermore, the Kappazunder car is visible as a masked area, especially in front-, left-, and right-facing cameras:

{{< image src="images/stadtpark_three_views.png" alt="Stadtpark three views" caption="Three views of the Stadtpark area with the Kappazunder car visible in front, left, and right-facing cameras." >}}

### Car Hood Masking

First reconstructions left much to be desired, and I started to implement [masking](../pipeline/masking).
Here's a comparison of some angles before (left) and after (right) masking:

{{< image src="images/stadtpark_before_after_yolo.png" alt="Stadtpark before and after masking and alignment" caption="Comparison of Stadtpark reconstruction before (left) and after (right) masking and alignment." >}}

### Manual Alignment

 Later, an additional manual alignment step was performed (first with RealityScan, in [later scenes](./kolonitzpark.md) directly inside Spirula Studio):

 {{< image src="images/2026-08-25_Model_cleaned.jpg" alt="Stadtpark cleaned model" caption="Cleaned model of the Stadtpark area after manual alignment." >}}

 Note that the image is cleaned of many floaters, but the fine details are already sharper. 60k steps were used; the floor is still very patchy, which becomes visible when moving above the camera height.


 ## Implementing Depth and Normal Maps

 Spirula Studio makes it easy to use depth and normal maps.
 They can be calculated directly from within the program using, e.g., [MoGe-2](https://github.com/microsoft/MoGe).

---

## Next Steps

- [Data Access](./../setup/data-access) — request Kappazunder data from Vienna
- [Quick Start](../getting-started/quick-start) — run your first reconstruction
- [Pipeline Overview](./../pipeline/overview) — understand the end-to-end workflow