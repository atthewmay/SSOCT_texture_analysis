# Segmentation algorithm: panel-by-panel explanation

## Narrative

The current RPE segmentation algorithm is easiest to understand as a sequence of increasingly local searches:

1. Find the ILM.
2. Estimate a coarse RPE-adjacent guide using a globally smooth hypersmoother path.
3. Flatten the image to that guide.
4. Build a lower-resolution boundary-enhancement image.
5. Run dynamic programming to obtain an initial globally smooth RPE estimate.
6. Return the estimate to original coordinates.
7. Build high-resolution local differential images near the RPE complex.
8. Run paired-surface DP variants for the original, choroidal-oriented, and EZ-oriented proposals.
9. Vertically refine/align proposal lines.
10. Export compact 1D lines for downstream flattening, en-face projection, and texture analysis.


# Segmentation algorithm panels

## 01. Raw B-scan

<p align="center"><img src="readme_panels/seg_01_01_raw_bscan.png" width="90%"></p>

Starting OCT B-scan. The ILM line is shown when available. This is the input carried by `ILMContext.original_image` and `RPEContext.original_image`.

**Code references:** `sp.process_bscan_1_3_26`, `ssf.ILMContext`, `ssf.RPEContext`

## 02. Coarse hypersmoother guide

<p align="center"><img src="readme_panels/seg_02_02_hypersmoother_path.png" width="90%"></p>

`step_rpe_hypersmoother_3_7_26` finds a coarse, globally smooth RPE-adjacent path. The path is used to flatten the B-scan before the more local/high-resolution stages.

**Code references:** `ssf.step_rpe_hypersmoother_3_7_26`, `suf.rpe_hypersmoother_DP_3_7_26`

## 03. Coarse hypersmoother image

<p align="center"><img src="readme_panels/seg_03_03_coarse_hypersmoothed_image.png" width="90%"></p>

The low-resolution image/cost surface used for the coarse guide. This is useful for explaining why the guide prefers the broad RPE/choroid complex.

**Code references:** `ssf.step_rpe_hypersmoother_3_7_26`

## 04. B-scan flattened to coarse guide

<p align="center"><img src="readme_panels/seg_04_04_flattened_to_hypersmoother.png" width="90%"></p>

The B-scan after warping to the coarse guide. This makes the later search region more stable across columns and reduces the effect of broad curvature.

**Code references:** `flattening_utility_functions.flatten_to_path`, `ssf.step_rpe_hypersmoother_3_7_26`

## 05. Downsampled working image

<p align="center"><img src="readme_panels/seg_05_05_downsampled_working_image.png" width="90%"></p>

`step_rpe_downsample_and_preprocess` creates a smaller working image for the first RPE pass.

**Code references:** `ssf.step_rpe_downsample_and_preprocess`

## 06. Boundary enhancement

<p align="center"><img src="readme_panels/seg_06_06_boundary_enhancement.png" width="90%"></p>

`step_rpe_compute_enhancement2` builds an image that emphasizes the relevant axial transition near the RPE complex. Computes axial graident, blurs, and then suppresses peaks below the top (most anteriorly oriented) 2 peaks to reduce choroidal signal.

**Code references:** `ssf.step_rpe_compute_enhancement2`, `suf.peakSuppressor`

## 07. Low-resolution DP cost

<p align="center"><img src="readme_panels/seg_07_07_lowres_dp_cost.png" width="90%"></p>

`step_rpe_DP_on_enh_2` runs a globally optimized dynamic-programming path through the enhanced image.

**Code references:** `ssf.step_rpe_DP_on_enh_2`

## 08. Low-resolution RPE path on raw image

<p align="center"><img src="readme_panels/seg_08_08_lowres_rpe_on_raw.png" width="90%"></p>

After `step_rpe_upsample` and `step_rpe_unsmooth`, the coarse RPE estimate is returned to the original B-scan coordinate space.

**Code references:** `ssf.step_rpe_upsample`, `ssf.step_rpe_unsmooth`

## 09. High-resolution differential image

<p align="center"><img src="readme_panels/seg_09_09_highres_diff_image.png" width="90%"></p>

`step_rpe_highres_diff_enh` builds a local high-resolution differential/gradient image around the RPE complex.

**Code references:** `ssf.step_rpe_highres_smooth`, `ssf.step_rpe_highres_diff_enh`

## 10. Lower-edge candidate image

<p align="center"><img src="readme_panels/seg_10_10_lower_edge_of_tubed.png" width="90%"></p>

A high-resolution image with regions far from RPE proposal suppressed, highlighting the lower edge used by later DP refinement.

**Code references:** `ssf.step_rpe_highres_diff_enh`

## 11. Original two-layer DP proposal

<p align="center"><img src="readme_panels/seg_11_11_original_two_layer_dp.png" width="90%"></p>

`step_rpe_highres_DP_two_layer` estimates a paired-surface proposal in the high-resolution band.

**Code references:** `ssf.step_rpe_highres_DP_two_layer`, `two_surface_utils`

## 12. Choroidal-oriented proposal

<p align="center"><img src="readme_panels/seg_12_12_choroidal_two_layer_dp.png" width="90%"></p>

`step_rpe_highres_DP_two_layer_choroidal` reruns/refines the RPE paired-surface logic with a focus on images with high choroidal signal.

**Code references:** `ssf.step_rpe_highres_DP_two_layer_choroidal`

## 13. EZ-oriented proposal

<p align="center"><img src="readme_panels/seg_13_13_ez_two_layer_dp.png" width="90%"></p>

`step_rpe_highres_DP_two_layer_EZ` preserves a paired-surface RPE proposal for images with robust EZ band.

**Code references:** `ssf.step_rpe_highres_DP_two_layer_EZ`

## 14. Vertical-shift refinement

<p align="center"><img src="readme_panels/seg_14_14_vertical_shift_refinement.png" width="90%"></p>

`step_rpe_vertical_shift_refine` aligns/refines the proposal lines vertically before export.

**Code references:** `ssf.step_rpe_vertical_shift_refine`

## 15. Final exported lines

<p align="center"><img src="readme_panels/seg_15_15_final_exported_lines.png" width="90%"></p>

Final compact set of paths saved into the segmentation `.npz` outputs for downstream flattening and texture projection.

**Code references:** `setup_data/02_segment_ILM_RPE.py::extract_lite`