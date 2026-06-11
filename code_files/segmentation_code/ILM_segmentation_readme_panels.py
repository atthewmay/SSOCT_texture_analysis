from __future__ import annotations

"""
Export ILM segmentation debug panels as individual cropped PNGs.

This mirrors ``segmentation_readme_panels.py`` for the RPE pipeline, but keeps the
panel set focused on the ILM pathway used by ``ILM_STEPS_2_28``.

Typical endpoint-step use:

    from code_files.segmentation_code.ILM_segmentation_readme_panels import (
        save_ilm_segmentation_readme_panels,
    )

    save_ilm_segmentation_readme_panels(
        ctx_ilm=ctx,
        out_dir="./docs/segmentation/readme_panels/ilm/",
        prefix="ilm",
    )

The saved PNGs are title-free, matching the existing RPE README panel exporter.
Panel letters/titles are written into ``ilm_panels_manifest.md`` and filenames.
"""

from pathlib import Path
from typing import Optional

import numpy as np

import code_files.segmentation_code.flattening_utility_functions as fuf
import code_files.segmentation_code.segmentation_plot_utils as spu
import code_files.segmentation_code.segmentation_utility_functions as suf
from code_files.segmentation_code.segmentation_readme_panels import (
    ReadmePanel,
    _get,
    _line_dict,
    _valid_array,
    save_one_panel_png,
    write_manifest_md,
)


# -----------------------------
# Coordinate helpers
# -----------------------------

def _as_1d_float(line) -> Optional[np.ndarray]:
    if line is None:
        return None
    y = np.asarray(line, dtype=float)
    if y.ndim != 1 or y.size == 0:
        return None
    return y


def _resample_line_to_shape(
    line,
    *,
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
) -> Optional[np.ndarray]:
    """Resample a 1D y(x) line from source image coordinates to target image coordinates."""
    y = _as_1d_float(line)
    if y is None:
        return None

    h_src, w_src = source_shape[:2]
    h_tgt, w_tgt = target_shape[:2]
    if h_src <= 0 or w_src <= 0 or h_tgt <= 0 or w_tgt <= 0:
        return None

    # Treat the line as spanning the horizontal extent of ``source_shape``.
    # If ``y.size == w_src`` this is simply 0, 1, ..., w_src - 1.
    # If not, this still makes the coordinate mapping explicit rather than
    # silently treating line length itself as the source image width.
    x_src = np.linspace(0, float(w_src - 1), int(y.size))
    x_tgt = np.linspace(0, float(w_src - 1), int(w_tgt))

    finite = np.isfinite(y)
    if finite.sum() < 2:
        return None

    y_interp = np.interp(x_tgt, x_src[finite], y[finite])
    return y_interp * (float(h_tgt) / float(h_src))


def _original_line_to_flat_original(ctx_ilm, line) -> Optional[np.ndarray]:
    """Map a final original-space ILM line back into the hypersmoother-flattened image."""
    y = _as_1d_float(line)
    if y is None:
        return None

    shift = _get(ctx_ilm, "hypersmoother_params.hypersmoother_shift_y_full")
    if shift is None:
        return y

    return fuf.warp_line_by_shift(y, shift, direction="to_flat")


def _original_line_to_original_panel(ctx_ilm, line, arr) -> Optional[np.ndarray]:
    """
    Map an original-space line to an original-space panel.

    This is intentionally explicit. It should only be used for panels whose
    background is in original B-scan coordinates. It does not try to infer
    coordinate space from line length or y-range.
    """
    y = _as_1d_float(line)
    if y is None or arr is None:
        return None

    arr = np.asarray(arr)
    orig = _get(ctx_ilm, "original_image")
    if orig is None or arr.ndim < 2:
        return None

    return _resample_line_to_shape(
        y,
        source_shape=np.asarray(orig).shape[:2],
        target_shape=arr.shape[:2],
    )


def _original_line_to_flat_working_panel(ctx_ilm, line, arr) -> Optional[np.ndarray]:
    """
    Map a final original-space ILM line onto a flattened/downsampled ILM panel.

    The README exporter is normally called after ``step_ilm_unsmooth``. At that
    point ``ilm_raw`` and ``ilm_smooth`` are original-space lines, while panels
    such as ``thinline_inv_cost``, ``final_DP_darkness_barrier_img``, and
    ``DP_refining_tube`` remain in the flattened/downsampled working image space.

    Transform is deliberately fixed, not guessed:
        original-space line -> hypersmoother-flattened original-space line
        -> resampled to the target working-panel shape.
    """
    y = _as_1d_float(line)
    if y is None or arr is None:
        return None

    arr = np.asarray(arr)
    orig = _get(ctx_ilm, "original_image")
    if orig is None or arr.ndim < 2:
        return None

    y_flat = _original_line_to_flat_original(ctx_ilm, y)
    return _resample_line_to_shape(
        y_flat,
        source_shape=np.asarray(orig).shape[:2],
        target_shape=arr.shape[:2],
    )


# -----------------------------
# Peak-suppression panel helpers
# -----------------------------

def _ilm_peak_suppression_debug(ctx_ilm):
    """
    Recreate the peak list and peak-derived suppression guide used by
    ``step_ilm_peak_suppression``.

    This intentionally mirrors the default config inside
    ``suf.peakSuppressor.peak_suppression_pipeline`` with
    ``use_third_peak=False`` plus the explicit ILM ``suppression_factor=0.1``.
    """
    enh = _get(ctx_ilm, "enh")
    if enh is None:
        return None, None

    _, peaks, _ = suf.peakSuppressor.extract_smoothed_and_peaks(
        enh,
        sigma=2.0,
        peak_prominence=0.02,
        peak_distance=20,
        ilm_line=None,
        min_offset=-15,
    )
    peak_line = suf.peakSuppressor._peaks_to_line(
        peaks,
        W=enh.shape[1],
        mode="top_unless_not_top2_intensity",
        peak_source_image=enh,
    )
    return peaks, peak_line


def _ilm_peak_overlay(ctx_ilm):
    """Overlay pre-suppression peak detections on the enhancement image."""
    img = _get(ctx_ilm, "enh")
    if img is None:
        return None

    peaks, _ = _ilm_peak_suppression_debug(ctx_ilm)
    if peaks is None:
        return img
    return spu.overlay_peaks_on_image(img, peaks)


def _ilm_peak_line(ctx_ilm, _rpe=None):
    _, peak_line = _ilm_peak_suppression_debug(ctx_ilm)
    return peak_line


# -----------------------------
# Panel definitions
# -----------------------------

def default_ilm_segmentation_readme_panels() -> list[ReadmePanel]:
    """Panels matching the current ILM pipeline narrative."""
    return [
        ReadmePanel(
            slug="panel_A_raw_bscan",
            title="Panel A. Raw B-scan",
            description=(
                "Starting OCT B-scan before ILM processing. This is "
                "`ILMContext.original_image`."
            ),
            source_functions=("ssf.ILMContext", "sp.process_bscan_1_3_26"),
            get_array=lambda ilm, rpe: _get(ilm, "original_image"),
        ),
        ReadmePanel(
            slug="panel_B_coarse_hypersmoother_guide",
            title="Panel B. Coarse hypersmoother guide",
            description=(
                "`step_ilm_hypersmoother` downsamples/blurs the raw B-scan and runs a first DP pass "
                "through image intensity to obtain a rough retinal-location guide. The guide is shown "
                "on the original B-scan because it is used to flatten the original image."
            ),
            source_functions=("ssf.step_ilm_hypersmoother", "suf.rpe_hypersmoother_DP"),
            get_array=lambda ilm, rpe: _get(ilm, "hypersmoother_params.coarse_hypersmoothed_img"),
            get_lines=lambda ilm, rpe: _line_dict(
                    hypersmoothed=_get(ilm, "hypersmoother_params.hypersmoother_y_dp"),
            ),
        ),
        ReadmePanel(
            slug="panel_C_flattened_downsampled_blurred",
            title="Panel C. Flattened, downsampled, blurred B-scan",
            description=(
                "The original B-scan is flattened to the coarse guide, mildly downsampled, and Gaussian blurred. "
                "This is the working image used by the subsequent ILM enhancement steps."
            ),
            source_functions=(
                "flattening_utility_functions.flatten_to_path",
                "ssf.step_ilm_downsample_and_preprocess",
            ),
            get_array=lambda ilm, rpe: _get(ilm, "img"),
        ),
        ReadmePanel(
            slug="panel_D_filtered_gradient_image",
            title="Panel D. Filtered vertical-gradient image",
            description=(
                "`step_ilm_compute_enhancement` computes upgoing/downgoing axial-gradient responses and "
                "subtracts an anisotropically blurred component to reduce deeper sclero-choroidal signal. "
                "The resulting filtered image is `ctx.enh`."
            ),
            source_functions=("ssf.step_ilm_compute_enhancement", "suf.compute_enh_diff"),
            get_array=lambda ilm, rpe: _get(ilm, "enh"),
        ),
        ReadmePanel(
            slug="panel_E_peak_detection_pre_suppression",
            title="Panel E. Peak detection before suppression",
            description=(
                "Column-wise peaks are identified with the same peak settings used by "
                "`step_ilm_peak_suppression`, but are shown here on the incoming filtered "
                "gradient image before suppression is applied. Peaks are overlaid in red."
            ),
            source_functions=("ssf.step_ilm_peak_suppression", "suf.peakSuppressor"),
            get_array=lambda ilm, rpe: _ilm_peak_overlay(ilm),
        ),
        ReadmePanel(
            slug="panel_F_peak_detection_post_suppression",
            title="Panel F. Peak detection after suppression",
            description=(""
            ),
            source_functions=("ssf.step_ilm_peak_suppression", "suf.peakSuppressor"),
            get_array=lambda ilm, rpe: _get(ilm,"peak_suppressed"),
        ),

        ReadmePanel(
            slug="panel_G_thinned_dp_input",
            title="Panel G. Thinned ILM DP input",
            description=(
                "`step_ilm_ax_grad_thinner` applies an additional vertical-gradient operation to the "
                "peak-suppressed image, increasing separation between ILM and residual deeper bands. "
                "The raw DP path from `step_ilm_DP` is overlaid in flattened working coordinates."
            ),
            source_functions=("ssf.step_ilm_ax_grad_thinner", "ssf.step_ilm_DP"),
            get_array=lambda ilm, rpe: _get(ilm, "thinline_inv_cost"),
            get_lines=lambda ilm, rpe: _line_dict(
                ilm_raw=_original_line_to_flat_working_panel(ilm, _get(ilm, "ilm_raw"), _get(ilm, "thinline_inv_cost")),
            ),
        ),
        ReadmePanel(
            slug="panel_H_darkness_barrier",
            title="Panel H. Darkness-barrier transition penalty",
            description=(
                "The modified DP pass penalizes large vertical jumps across low-intensity/high-cost regions. "
                "This panel shows the darkness-barrier image used by `run_DP_on_cost_matrix`, with the raw "
                "ILM DP path overlaid."
            ),
            source_functions=(
                "ssf.step_ilm_DP",
                "suf.calculate_darkness_barrier_and_Bcs",
                "suf.run_DP_on_cost_matrix",
            ),
            get_array=lambda ilm, rpe: _get(ilm, "final_DP_darkness_barrier_img"),
            get_lines=lambda ilm, rpe: _line_dict(
                ilm_raw=_original_line_to_flat_working_panel(
                    ilm,
                    _get(ilm, "ilm_raw"),
                    _get(ilm, "final_DP_darkness_barrier_img"),
                ),
            ),
        ),
        ReadmePanel(
            slug="panel_I_refining_dp_tube",
            title="Panel I. Constrained refining DP pass",
            description=(
                "Because gradient operations can shift the apparent height of a boundary, "
                "`step_ilm_DP_refiner` returns to the pre-thinned intensity image and constrains a second DP "
                "pass to a tube around the raw ILM path. Both the raw and refined ILM paths are shown."
            ),
            source_functions=("ssf.step_ilm_DP_refiner", "suf.apply_gaussian_tube_mul"),
            get_array=lambda ilm, rpe: _get(ilm, "DP_refining_tube"),
            get_lines=lambda ilm, rpe: _line_dict(
                ilm_raw=_original_line_to_flat_working_panel(ilm, _get(ilm, "ilm_raw"), _get(ilm, "DP_refining_tube")),
                ilm_smooth=_original_line_to_flat_working_panel(ilm, _get(ilm, "ilm_smooth"), _get(ilm, "DP_refining_tube")),
            ),
        ),
        ReadmePanel(
            slug="panel_J_final_ilm_original_space",
            title="Panel J. Final ILM segmentation in original coordinates",
            description=(
                "After upsampling and unflattening, the final ILM segmentation is returned to original B-scan "
                "coordinates. `ilm_raw` is the first DP result and `ilm_smooth` is the refined final line."
            ),
            source_functions=("ssf.step_ilm_upsample", "ssf.step_ilm_unsmooth"),
            get_array=lambda ilm, rpe: _get(ilm, "original_image"),
            get_lines=lambda ilm, rpe: _line_dict(
                # ilm_raw=_original_line_to_original_panel(ilm, _get(ilm, "ilm_raw"), _get(ilm, "original_image")),
                ilm_smooth=_original_line_to_original_panel(ilm, _get(ilm, "ilm_smooth"), _get(ilm, "original_image")),
            ),
        ),
    ]


# -----------------------------
# Export function / optional pipeline step
# -----------------------------

def save_ilm_segmentation_readme_panels(
    ctx_ilm,
    *,
    out_dir: str | Path,
    prefix: str = "ilm",
    panels: Optional[list[ReadmePanel]] = None,
    dpi: int = 300,
    transparent: bool = True,
    write_manifest: bool = True,
) -> list[dict]:
    """Save ILM README/paper panels and return manifest rows."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    panels = panels or default_ilm_segmentation_readme_panels()

    rows: list[dict] = []
    for idx, panel in enumerate(panels, start=1):
        arr = panel.get_array(ctx_ilm, None)
        if not _valid_array(arr):
            print(f"Skipping missing/non-image ILM panel: {panel.slug}")
            continue

        filename = f"{prefix}_{idx:02d}_{panel.slug}.png"
        out_path = out_dir / filename

        lines = panel.get_lines(ctx_ilm, None)
        save_one_panel_png(
            arr,
            out_path=out_path,
            lines=lines,
            dpi=dpi,
            transparent=transparent,
        )

        rows.append(
            {
                "index": idx,
                "slug": panel.slug,
                "title": panel.title,
                "description": panel.description,
                "source_functions": panel.source_functions,
                "filename": str(out_path),
            }
        )

    if write_manifest:
        write_manifest_md(
            rows,
            out_dir / "ilm_panels_manifest.md",
            title="ILM segmentation algorithm panels",
        )

    return rows


def step_ilm_save_readme_panels(ctx_ilm):
    """Optional endpoint step if you want to import this function directly in a debug pipeline."""
    save_ilm_segmentation_readme_panels(
        ctx_ilm=ctx_ilm,
        out_dir="/Volumes/T9/iowa_research/Han_AIR_Dec_2025/results/ILM_segmentation_panels/",
        prefix="ilm",
    )
    return ctx_ilm