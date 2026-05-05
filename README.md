## Visual Overview

### Volume Processing

| Raw OCT volume | RPE/ILM segmentation |
|---|---|
| <img src="docs/assets/raw_scroll.gif" width="420"> | <img src="docs/assets/layer_scroll.gif" width="420"> |

| RPE-flattened volume | Texture volume (local binary pattern entropy) |
|---|---|
| <img src="docs/assets/flat_scroll.gif" width="420"> | <img src="docs/assets/texture_scroll.gif" width="420"> |

### En-face Outputs

<p align="center">
  <img src="docs/assets/texture_volume_to_enface.png" width="70%">
  <em> 3D to 2D texture projection – 3D texture volume slabs are mean-projected to form an en face computed texture, which is then registered by rotation to align fovea and optic nerve head along the horizontal. Left eye is mirrored to match right eye. </em>
</p>

<p align="center">
  <img src="docs/assets/slab_mean_computed_texture.png" width="70%">
  <em>Slab-mean texture calculation – The mean of the outer-retinal slab is obtained, and the textures are calculated directly on the mean image and registered.</em>
</p>

### Development Tools

<p align="center">
  <img src="docs/assets/annotation_tool.gif" width="70%">
  <br>
  <em>Interactive fovea / optic nerve head annotation tool.</em>
</p>

- Napari based tools for inspecting volumes, layers, and textures
    - Interactive for prototyping segmentations and debugging
- Annotation tools for data preprocessing (optic nerve head and fovea selection)


The license applies to source code in this repository. It does not grant rights to any clinical imaging data, annotations, trained models, or third-party datasets referenced by the code.
