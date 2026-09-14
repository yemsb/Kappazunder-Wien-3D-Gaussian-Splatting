---
title: Showcases
layout: landing
menu:
  after:
    weight: 5
---

# Project Showcases

See what you can create with this pipeline - real reconstructions from Vienna's Kappazunder data.

## Stadtpark Reconstruction

A complete 3D Gaussian Splatting reconstruction of Vienna's Stadtpark using the full pipeline.

{{% hint info %}}
See [Stadtpark Walkthrough](../docs/captures/stadtpark) for details.
{{% /hint %}}

{{< video src="https://github.com/yemsb/Kappazunder-Wien-3D-Gaussian-Splatting/releases/download/assets/2026-09-08_20-08-56.mp4" >}}

*Flythrough of the trained Stadtpark model showing detailed trees, benches, a statue, and park paths.*

### Technical Details

- **Data Source**: Kappazunder trajectory 15767 (`Los_6A`) (Stadtpark area)
- **Processing**: Full pipeline with AoI filtering, manual alignment step, vehicle masking, and depth/normal maps
- **Training**: 60k iterations in Spirula Studio
- **Final Model**: 2M Gaussians, sherical harmonic order 3

## Kolonitzplatz Example

Reconstruction of a Vienna square showing the pipeline's handling of complex urban scenes.

{{% hint info %}}
See [Kolonitzplatz Walkthrough](../docs/captures/kolonitzpark) for details.
{{% /hint %}}

{{< video src="https://github.com/yemsb/Kappazunder-Wien-3D-Gaussian-Splatting/releases/download/assets/2026-09-14_kolonitzplatz_render_1080p.mp4" >}}

*This video is also available on [YouTube](https://youtu.be/ajA4t0YyMUI) in 4K resolution.*

### Technical Details

- **Data Source**: Kappazunder trajectory 15760, 20727 (`Los_6B`) (Kolonitzplatz area)
- **Processing**: Full pipeline with AoI filtering, manual alignment step, vehicle & person masking, and depth/normal maps
- **Training**: 100k iterations in Spirula Studio
- **Final Model**: 6M Gaussians, sherical harmonic order 3

## Try It Yourself

All showcases were created using the exact pipeline documented in this repository. You can reproduce similar results by:

1. Requesting Kappazunder data for your area of interest via [Vienna's geoportal](https://geoportal.wien.gv.at/map/main/geodaten/)
2. Following the [Getting Started](../docs/getting-started/quick-start) guide
3. Using the provided configuration examples as starting points
4. Training in Spirula Studio or Lichtfeld Studio

## Contribute Your Results

Have you created an impressive reconstruction with this pipeline? Consider sharing it:
- Open an issue with a link to your model or video
- Add your results to the project wiki