---
title: Kolonitzpark Capture Walkthrough
weight: 30
bookToC: true
---

# Kolonitzpark Capture Walkthrough

The Kolonitzplatz (named after archbishop Leopold Karl Graf Kollonitsch) is a public square in Vienna featuring a large neogothic church, [St. Othmar](https://st-othmar.at/150-jahre-st-othmar/) (built between 1866-1873), a small park area, and an adjacent [primary school](https://www.geschichtewiki.wien.gv.at/VS_Kolonitzgasse_15) (built [between 1870–1872](https://www.architektenlexikon.at/de/1101.htm)).

## Overview

The Kolonitzplatz is a public square in Vienna featuring:
- A large church ([St. Othmar](https://st-othmar.at/150-jahre-st-othmar/))
- A small park area
- An adjacent primary school ([VS Kolonitzgasse 15](https://www.geschichtewiki.wien.gv.at/VS_Kolonitzgasse_15))
- Mixed seasons (summer and autumn)
- Mixed lighting conditions

{{< image src="images/front_facing-ezgif.com-optimize.gif">}}

The immediate area surrounding the Andreas Zelinka monument was chosen from the Kappazunder dataset for this reconstruction. 

{{< hint warning >}}
Easy to medium difficulty for this reconstruction, as the main street is captured from two different lanes, but there are many transient objects (cars, people). 
Again, the calibration is not perfect, which can be mitigated by running an alignment step.
{{< /hint >}}

### Area of Interest (AoI)
```
# EPSG: 31256
# x_m		y_m
4278.4555	341299.8990
4336.3397	341359.7583
4360.2028	341320.4970
4364.8786	341312.7576
4379.3094	341291.5549
4392.4502	341274.7863
4411.9599	341251.1650
4441.0834	341218.4742
4435.1781	341212.6293
4416.2327	341197.7149
4410.6700	341191.8297
4402.2857	341166.9186
4395.5137	341166.6767
4377.4551	341187.8794
4365.2817	341202.3101
4356.3734	341213.1130
4334.8079	341238.0645
4334.0017	341239.5962
4322.1105	341252.8580
4311.7913	341265.3539
4307.1557	341270.8359
4288.3313	341293.0867
4278.4555	341299.8990
```

{{< image src="images/kolonitzplatz_selection_preview.jpg" alt="Stadtpark Area of Interest" caption="Area of Interest (AoI) for the Stadtpark reconstruction." >}}

After [frustum selection](../pipeline/camera-selection/#2-sensor-roles--frustum-geometry) and an acceptance border of 10 meters around the AoI, 1051 images (~4 GB) were selected.

## Reconstruction

### The Importance of Recalibration and Depth/Normal Maps

What I gathered from this reconstruction more than from the previous Stadtpark reconstruction is that *the single greatest improvement is made from **recalibration** of the camera poses and using **depth/normal maps***.
This time, I first ran the optimised pipeline (using realingment and depth/normal maps directly generated in Spirula Studio).

{{< image src="images/kolonitzplatz_SS_20260910-110222.jpg" alt="Kolonitzpark reconstruction" caption="Kolonitzpark reconstruction using Spirula Studio with realignment and depth/normal maps." >}}

{{< image src="images/kolonitzplatz_details_1.png" >}}

{{< image src="images/kolonitzplatz_details_2.png" >}}

{{< image src="images/kolonitzplatz_details_3.png" >}}

*Note that cars and people were masked in this reconstruction.*

### Raw Calibration Results

By contrast, if using the original calibration and depth/normal maps, the fit **was not stable** and the reconstruction dissolved after ~5k steps.
When using the original calibration and no depth/normal maps, the fit was stable but the reconstruction was very poor:

{{< image src="images/kolonitzplatz_20260912-184321.jpg" alt="Kolonitzpark reconstruction" caption="Kolonitzpark reconstruction using Spirula Studio without realignment and depth/normal maps." >}}

{{< image src="images/kolonitzplatz_original_details_1.png" >}}

{{< image src="images/kolonitzplatz_original_details_2.png" >}}

{{< image src="images/kolonitzplatz_original_details_3.png" >}}

Keep in mind that the second model (i.e. using the original calibration) even had the advantage of using the initial LiDAR point cloud, while the first model (i.e. using the optimised pipeline) used SfM-generated points.
Furthermore, the second model made use of the [sky dome](../pipeline/lidar-conversion/#26-adding-a-sky-dome) and the point cloud was augmented with the city's [rooftop LiDAR data (DOM)](https://www.wien.gv.at/stadtplanung/digitales-oberflaechenmodell).

Both models were trained for the same number of steps and with the same number of Gaussians.


### Model Availability

The first shown model consists of 6M Gaussians and was trained for 100k steps.
It is shown in the [showcase video](../../showcases/_index/#kolonitzplatz-example) and can be explored at [SuperSplat](https://superspl.at/scene/351db14a).

---

## Next Steps

- [Data Access](./../setup/data-access) — request Kappazunder data from Vienna
- [Quick Start](../getting-started/quick-start) — run your first reconstruction
- [Pipeline Overview](./../pipeline/overview) — understand the end-to-end workflow