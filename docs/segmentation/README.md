# Segmentation pipeline

[Detailed pictorial algorithm explanation](docs/segmentation/segmentation_algorithm.md)

This folder documents the OCT B-scan layer segmentation code used in this project. The goal is to turn each raw SSOCT B-scan into column-wise retinal layer paths that can be stacked across `z` and reused by the flattening, en-face, and texture-analysis pipelines.

<p align="center">
  <img src="../assets/raw_scroll.gif" width="44%">
  <img src="../assets/layer_scroll.gif" width="44%">
</p>

<p align="center">
  <em>Raw OCT volume and the corresponding layer-overlay scroll view.</em>
</p>


---

## What this code produces

For each processed volume, the segmentation runner writes:

```text
<outputs_root>/
  <volume_id>/
    z0000.npz
    z0025.npz
    ...
    <volume_id>_stacked.npz
```

Each per-slice `.npz` contains lightweight 1D paths and key intermediate lines. The stacked file contains arrays shaped approximately:

```text
(n_processed_z_slices, image_width)
```

The most important exported paths are:

| Key | Meaning |
|---|---|
| `ilm_smooth` | Final smoothed ILM line. |
| `hypersmoother_path` | Coarse RPE-adjacent guide line used to flatten the B-scan for later processing. |
| `rpe_smooth` | Penultimate / coarse RPE output retained for debugging and comparison. |
| `original_method_y1_rescaled`, `original_method_y2_rescaled` | Two-layer high-resolution proposal from the original RPE-oriented path (y2 is RPE). |
| `choroidal_method_y1_rescaled`, `choroidal_method_y2_rescaled` | Choroidal-oriented two-layer proposal (y1 is RPE). |
| `EZ_method_y1_rescaled`, `EZ_method_y2_rescaled` | EZ-oriented two-layer proposal (y2 is RPE). |
| `*_vertical_shifted` | Vertically shifted/refined versions used by the latest unified path. |

For downstream texture work, the important point is that this segmentation output is a compact layer package: it is not the raw OCT volume, and it does not need to preserve every temporary image used during debugging.

---

## Quick start: segment one volume

A small single-volume demo runner is included as:

```text
code_files/setup_data/demo_segment_single_volume.py
```

Prior to running, user needs to label the optic nerve head and fovea with our included tool, paired_annotator. 

<p align="center">
  <img src="../assets/annotation_tool.gif" width="90%">
</p>

Example:

```bash
python code_files/setup_data/demo_segment_single_volume.py \
  --volume "/path/to/volume_name.img" \
  --annotation-root "/path/to/annotations_root" \
  --outputs-root "/path/to/layers_out" \
  --z-step 25 \ # set this to 1 for actual volume segmentation
  --pipeline latest \
  --max-workers 8
```

For a fast smoke test:

```bash
python code_files/setup_data/demo_segment_single_volume.py \
  --volume "/path/to/volume.img" \
  --annotation-root "/path/to/annotations_root" \
  --outputs-root "/tmp/seg_demo" \
  --z-step 500 \
  --pipeline latest \
  --max-workers 3
```

`--z-step` controls the resolution in the `z` direction. `--z-step 1` processes every B-scan. `--z-step 25` processes every 25th B-scan. Larger values are needed for high-res texture analysis.

Available RPE pipelines in the demo script: Just use the latest. 
- Users should extend this project by creating their own pipeline objects. 
- The pipelines file acts like an additional table of ccontents explaining the segmentation algorithm. 

| CLI value | Pipeline object |
|---|---|
| `latest` | `sp.RPE_STEPS_unified_3_19_26` |

The default ILM pipeline is:

```python
sp.ILM_STEPS_2_28
```

---
## Interactive Development

A real gem of this project is the interactivity. 

Use the file napari_run_with_layers_pagination.py for a smooth-scrolling visualization of volumes and layers (and textures if relevant). Buttons on the right hand side allow for testing of new pipelines while viewing (kick off a segmentation job as background process and display a new arrayboard saved as a pdf).

## How the segmentation code is organized

The segmentation code is intentionally split into three layers:

```text
code_files/segmentation_code/
  segmentation_pipelines.py       # table of contents / current recipes
  segmentation_step_functions.py  # named state-mutating steps
  segmentation_utility_functions.py # helpers for the above
  segmentation_plot_utils.py # Useful for debugging and display
  flattening_utility_functions.py 
  two_surface_utils.py # Library for simultaneous layer proposal needed for most accurate RPE identification when other bands (choroidal vessels or EZ are prominant)
```

### `segmentation_pipelines.py`

This is the best starting point. Treat it as the table of contents for the segmentation project.

A pipeline is just a list of step functions:

```python
RPE_STEPS_unified_3_19_26 = [
    ssf.step_rpe_init_working,
    ssf.step_rpe_hypersmoother_3_7_26,
    ssf.step_rpe_downsample_and_preprocess,
    ssf.step_rpe_compute_enhancement2,
    ssf.step_rpe_DP_on_enh_2,
    ssf.step_rpe_upsample,
    ssf.step_rpe_unsmooth,
    ssf.step_rpe_highres_smooth,
    ssf.step_rpe_highres_diff_enh,
    ssf.step_rpe_highres_DP_two_layer,
    ssf.step_rpe_highres_DP_two_layer_choroidal,
    ssf.step_rpe_highres_DP_two_layer_EZ,
    ssf.step_rpe_highres_unsmooth,
    ssf.step_rpe_vertical_shift_refine,
]
```

When you want to try a new segmentation idea, add a new step function in `segmentation_step_functions.py`, then add a new named list in `segmentation_pipelines.py`. Do not overwrite the last working pipeline. The dated pipeline names are useful because they preserve the algorithm history and make it easy to reproduce old results.

### `segmentation_step_functions.py`

This file defines the state containers and the step functions.

Core objects:

| Object | Role |
|---|---|
| `ILMContext` | Carries the ILM image, intermediate maps, and final ILM paths through the ILM pipeline. |
| `RPEContext` | Carries the RPE image, intermediate maps, high-resolution context, two-layer DP outputs, and final paths through the RPE pipeline. |
| `HypersmootherParams` | Stores coarse guide paths and flattening shifts. |
| `HighResContext` | Stores higher-resolution gradient images and refined paths. |
| `twoLayerDPContext` | Stores paired-surface DP outputs for original/choroidal/EZ-oriented proposals. |

The step functions mutate the context and return it:

```python
def step_rpe_highres_diff_enh(ctx: RPEContext) -> RPEContext:
    ...
    ctx.highres_ctx.diff_down_up = diff_down_up
    return ctx
```

This pattern is intentionally simple: every step has access to previous outputs, and the final context can be plotted or reduced into lightweight arrays.

### `segmentation_utility_functions.py`

This file should hold reusable image-processing and dynamic-programming primitives: enhancement images, peak suppression, path upsampling, tube smoothers, one-surface DP, two-surface DP helpers, etc.

If a function is reusable and does not need the full `RPEContext` / `ILMContext`, it belongs here.

### `segmentation_plot_utils.py`

This is the visualization layer. `ArrayBoard` is the main lightweight debug workhorse:

```python
AB = spu.ArrayBoard(skip=False, plt_display=False, save_tag=f"_{ctx.ID}")
AB.add(ctx.original_image, lines={"rpe": ctx.rpe_smooth}, title="final RPE")
AB.render()
```

For the README, prefer exporting one panel per PNG with `save_segmentation_readme_panels(...)` instead of saving a giant ArrayBoard. This makes GitHub/mobile rendering much cleaner and lets text explain each panel.


---

## Developer notes

- Keep `segmentation_pipelines.py` as the canonical place to discover “what is current.”
- Keep old pipelines unless they are actively broken and misleading.
- Prefer new dated pipeline names over silent edits to old lists.
- Use `--z-step 500` and `--max-workers 1` for quick smoke tests.
- Use larger `--z-step` values before running all slices.
- Save README figures as cropped PNGs rather than giant multi-panel PDFs.
