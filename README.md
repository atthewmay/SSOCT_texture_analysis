## Overview
This repository contains: 
1. Novel, open-source RPE and ILM segmentation algorithms for swept source OCT imaging, robust across diverse retinal degenerative phenotypes
    - [Quickstart guide for segmentation algorithms](docs/segmentation/README.md)
    - [Detailed pictorial algorithm explanation](docs/segmentation/SEGMENTATION_ALGORITHM.md)
2. Retinal texture analysis methods
3. The specific ML pipeline used in our study to distinguish autoimmune retinopathy from retinitis pigmentosa

### Volume Processing

<!-- | Raw OCT volume | RPE/ILM segmentation |
|---|---|
| <img src="docs/assets/raw_scroll.gif" width="420"> | <img src="docs/assets/layer_scroll.gif" width="420"> |

| RPE-flattened volume | Texture volume (local binary pattern entropy) |
|---|---|
| <img src="docs/assets/flat_scroll.gif" width="420"> | <img src="docs/assets/texture_scroll.gif" width="420"> | -->

<p align="center">
  <strong>Raw OCT volume</strong><br>
  <img src="docs/assets/raw_scroll.gif" width="700">
</p>

<p align="center">
  <strong>Open Source RPE/ILM segmentation</strong><br>
  <img src="docs/assets/layer_scroll.gif" width="700">
</p>

<p align="center">
  <strong>RPE-flattened volume</strong><br>
  <img src="docs/assets/flat_scroll.gif" width="700">
</p>

<p align="center">
  <strong>Texture volume: local binary pattern entropy</strong><br>
  <img src="docs/assets/texture_scroll.gif" width="700">
</p>

### En-face Outputs

<p align="center">
  <img src="docs/assets/texture_volume_to_enface.png" width="700">
    <br>
  <em> 3D to 2D texture projection – 3D texture volume slabs are mean-projected to form an en face computed texture, which is then registered by rotation to align fovea and optic nerve head along the horizontal. Left eye is mirrored to match right eye. </em>

</p>

<p align="center">
  <img src="docs/assets/slab_mean_computed_texture.png" width="700">
    <br>
  <em>Slab-mean texture calculation – The mean of the outer-retinal slab is obtained, and the textures are calculated directly on the mean image and registered.</em>
</p>

### Development Tools
- Napari based tools for inspecting volumes, layers, and textures
    - Interactive for prototyping segmentations and debugging
- Annotation tools for data preprocessing (optic nerve head and fovea selection)

<p align="center">
  <img src="docs/assets/napari_demo.gif" width="700">
  <br>
  <em>Interactive viewer with prototyping/debugging tools.</em>
</p>


<p align="center">
  <img src="docs/assets/annotation_tool.gif" width="700">
  <br>
  <em>Interactive fovea / optic nerve head annotation tool.</em>
</p>

### Get Started



The license applies to source code in this repository. It does not grant rights to any clinical imaging data, annotations, trained models, or third-party datasets referenced by the code.

