from __future__ import annotations
# Reviewed

"""
Export segmentation debug panels as individual cropped PNGs.

This is intended to replace a giant README/ArrayBoard-style figure with
clean, title-free PNG panels plus a Markdown manifest explaining what each
panel shows.

Typical use inside a segmentation endpoint/debug step:

    from code_files.segmentation_code.segmentation_readme_panels import (
        save_segmentation_readme_panels,
    )

    save_segmentation_readme_panels(
        ctx_ilm=ctx.ilm_ctx,
        ctx_rpe=ctx,
        out_dir="docs/assets/segmentation_panels/example",
        prefix=ctx.ID,
        final_rpe_pathway="original",  # original, choroidal, or EZ
    )

The saved PNGs have no axes, titles, borders, or whitespace. The text belongs
in the generated `panels_manifest.md`.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import code_files.segmentation_code.segmentation_plot_utils as spu
import code_files.segmentation_code.segmentation_utility_functions as suf


ArrayGetter = Callable[[object, object], Optional[np.ndarray]]
LinesGetter = Callable[[object, object], dict[str, np.ndarray]]


@dataclass(frozen=True)
class ReadmePanel:
    slug: str
    title: str
    description: str
    source_functions: tuple[str, ...]
    get_array: ArrayGetter
    get_lines: LinesGetter = lambda ilm, rpe: {}


def _get(obj, dotted: str, default=None):
    """Safely read dotted attributes, e.g. _get(ctx, 'highres_ctx.diff_down_up')."""
    cur = obj
    for part in dotted.split("."):
        if cur is None:
            return default
        cur = getattr(cur, part, default)
    return cur


def _line_dict(**items):
    return {k: v for k, v in items.items() if v is not None}


def _valid_array(arr) -> bool:
    if arr is None:
        return False
    arr = np.asarray(arr)
    if arr.ndim < 2:
        return False
    if arr.size == 0:
        return False
    return True


def save_one_panel_png(
    arr: np.ndarray,
    *,
    out_path: Path,
    lines: Optional[dict[str, np.ndarray]] = None,
    dpi: int = 300,
    cmap: str = "gray",
    transparent: bool = True,
    line_width: float = 0.8,
) -> None:
    """Save one image panel with no title/axes/whitespace."""
    arr = np.asarray(arr)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Keep the existing square-panel export behavior.
    h, w = arr.shape[:2]
    fig_w = 6.0
    # fig_h = max(0.5, fig_w * h / max(w, 1))
    fig_h = 6.0

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi, frameon=False)
    ax = fig.add_axes([0, 0, 1, 1])

    ax.imshow(arr, cmap=cmap, aspect="auto")

    for name, line in (lines or {}).items():
        if line is None:
            continue
        line = np.asarray(line)
        if line.ndim != 1:
            continue

        if name not in spu.LAYER_STYLE:
            print(f"Couldn't find name {name} in spu.LAYER_STYLE dictionary, falling bak to default I think")
        else:
            print(f"did find name {name} in spu.LAYER_STYLE dictionary")
        style = spu.LAYER_STYLE.get(name, None)
        x = np.arange(line.shape[0])
        if style is not None:
            ax.plot(
                x,
                line,
                style.get("fmt", "-"),
                lw=style.get("lw",line_width),
                alpha=style.get("alpha", 0.9),
            )
        else:
            ax.plot(x, line, lw=line_width, alpha=0.9)

    ax.set_axis_off()
    fig.savefig(
        out_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0,
        transparent=transparent,
    )
    plt.close(fig)


# -----------------------------
# RPE peak-overlay helpers
# -----------------------------

def _rpe_peak_suppression_debug(ctx_rpe):
    """Recreate the peaks detected on the image entering RPE peak suppression.

    This mirrors the peak-detection defaults inside
    ``suf.peakSuppressor.peak_suppression_pipeline``. The exporter runs after
    ``step_rpe_unsmooth``, so the correctly flattened/downsampled ILM guide is
    recovered from the context history as ``flat_ilm_seg``.
    """
    enh = _get(ctx_rpe, "enh")
    if enh is None:
        enh = _get(ctx_rpe, "enh_f")
    if enh is None:
        return None

    enh = np.asarray(enh)
    ilm_line = _get(ctx_rpe, "flat_ilm_seg")
    if ilm_line is not None:
        ilm_line = np.asarray(ilm_line, dtype=float)
        if ilm_line.shape != (enh.shape[1],):
            # Do not guess a coordinate transform here. A mismatched line is
            # safer to omit than to overlay/use in the wrong image space.
            ilm_line = None

    _, peaks, _ = suf.peakSuppressor.extract_smoothed_and_peaks(
        enh,
        sigma=2.0,
        peak_prominence=0.02,
        peak_distance=20,
        ilm_line=ilm_line,
        min_offset=-15,
    )
    return peaks


def _rpe_peak_overlay(ctx_rpe):
    """Overlay pre-suppression RPE peak detections in red on the filtered image."""
    enh = _get(ctx_rpe, "enh")
    if enh is None:
        enh = _get(ctx_rpe, "enh_f")
    if enh is None:
        return None

    peaks = _rpe_peak_suppression_debug(ctx_rpe)
    if peaks is None:
        return enh
    return spu.overlay_peaks_on_image(enh, peaks)


# -----------------------------
# High-resolution gradient panel helpers
# -----------------------------

def _ensure_highres_gradient_images(ctx_rpe):
    """Return the high-resolution differential and its two blurred components.

    ``step_rpe_highres_diff_enh`` currently stores ``diff_down_up`` but leaves
    ``hblur_down`` and ``hblur_up`` as local variables. For README/paper export,
    reuse stored component images when available; otherwise reproduce the exact
    utility call on ``highres_smoothed_img`` and cache the components on
    ``highres_ctx``.
    """
    highres_ctx = _get(ctx_rpe, "highres_ctx")
    if highres_ctx is None:
        return None, None, None

    diff_down_up = _get(ctx_rpe, "highres_ctx.diff_down_up")
    hblur_down = _get(ctx_rpe, "highres_ctx.hblur_down")
    hblur_up = _get(ctx_rpe, "highres_ctx.hblur_up")
    if diff_down_up is not None and hblur_down is not None and hblur_up is not None:
        return diff_down_up, hblur_down, hblur_up

    img = _get(ctx_rpe, "highres_smoothed_img")
    if img is None:
        return diff_down_up, hblur_down, hblur_up

    def cfg_value(name, default):
        value = _get(ctx_rpe, f"highres_cfg.{name}")
        return default if value is None else value

    calculated_diff, _, calculated_hblur_down, calculated_hblur_up = (
        suf.diff_boundary_enhance_and_blur_horiz(
            img,
            down_hblur=cfg_value("down_hblur", 40),
            up_hblur=cfg_value("up_hblur", 50),
            down_vertical_kernel_size=cfg_value("down_vertical_kernel_size", 25),
            up_vertical_kernel_size=cfg_value("up_vertical_kernel_size", 15),
        )
    )

    if diff_down_up is None:
        diff_down_up = calculated_diff
    if hblur_down is None:
        hblur_down = calculated_hblur_down
        setattr(highres_ctx, "hblur_down", hblur_down)
    if hblur_up is None:
        hblur_up = calculated_hblur_up
        setattr(highres_ctx, "hblur_up", hblur_up)

    return diff_down_up, hblur_down, hblur_up


def _highres_diff_down_up(ctx_rpe):
    return _ensure_highres_gradient_images(ctx_rpe)[0]


def _highres_hblur_down(ctx_rpe):
    return _ensure_highres_gradient_images(ctx_rpe)[1]


def _highres_hblur_up(ctx_rpe):
    return _ensure_highres_gradient_images(ctx_rpe)[2]


# -----------------------------
# Final-path selection helper
# -----------------------------

def _normalize_final_rpe_pathway(pathway: str) -> str:
    key = str(pathway).strip().lower()
    aliases = {
        "original": "original",
        "choroidal": "choroidal",
        "ez": "EZ",
    }
    if key not in aliases:
        raise ValueError(
            "final_rpe_pathway must be one of: 'original', 'choroidal', or 'EZ'"
        )
    return aliases[key]


def _selected_final_rpe_line(ctx_rpe, pathway: str):
    """Return the final vertically shifted line for one selected RPE pathway."""
    pathway = _normalize_final_rpe_pathway(pathway)

    if pathway == "original":
        shifted = _get(ctx_rpe, "two_layer_dp_ctx.y2_vertical_shifted")
        assert shifted is not None
        return shifted # if shifted is not None else _get(ctx_rpe, "two_layer_dp_ctx.y2_rescaled")

    if pathway == "choroidal":
        shifted = _get(ctx_rpe, "two_layer_dp_ctx_choroidal.y1_vertical_shifted")
        assert shifted is not None
        return shifted # if shifted is not None else _get(ctx_rpe, "two_layer_dp_ctx_choroidal.y1_rescaled")

    shifted = _get(ctx_rpe, "two_layer_dp_ctx_EZ.y2_vertical_shifted")
    assert shifted is not None
    return shifted # if shifted is not None else _get(ctx_rpe, "two_layer_dp_ctx_EZ.y2_rescaled")


def default_segmentation_readme_panels(
    final_rpe_pathway: str = "original",
) -> list[ReadmePanel]:
    """Panels matching the current unified RPE pipeline narrative."""
    final_rpe_pathway = _normalize_final_rpe_pathway(final_rpe_pathway)

    return [
        ReadmePanel(
            slug="01_raw_bscan",
            title="Raw B-scan",
            description=(
                "Starting OCT B-scan. The ILM line is shown when available. "
                "This is the input carried by `ILMContext.original_image` and `RPEContext.original_image`."
            ),
            source_functions=("sp.process_bscan_1_3_26", "ssf.ILMContext", "ssf.RPEContext"),
            get_array=lambda ilm, rpe: _get(rpe, "original_image"),
            get_lines=lambda ilm, rpe: _line_dict(ilm_smooth=_get(ilm, "ilm_smooth")),
        ),
        ReadmePanel(
            slug="02_hypersmoother_path",
            title="Coarse hypersmoother guide",
            description=(
                "`step_rpe_hypersmoother_3_7_26` finds a coarse, globally smooth RPE-adjacent path. "
                "The path is used to flatten the B-scan before the more local/high-resolution stages."
            ),
            source_functions=("ssf.step_rpe_hypersmoother_3_7_26", "suf.rpe_hypersmoother_DP_3_7_26"),
            get_array=lambda ilm, rpe: _get(rpe, "original_image"),
            get_lines=lambda ilm, rpe: _line_dict(
                hypersmoothed=_get(rpe, "hypersmoother_params.hypersmoother_path"),
                ilm_smooth=_get(ilm, "ilm_smooth"),
            ),
        ),
        ReadmePanel(
            slug="03_coarse_hypersmoothed_image",
            title="Coarse hypersmoother image and DP guide",
            description=(
                "The low-resolution image/cost surface used for the coarse guide, with the DP path "
                "shown directly in the same coarse coordinate space."
            ),
            source_functions=("ssf.step_rpe_hypersmoother_3_7_26",),
            get_array=lambda ilm, rpe: _get(rpe, "hypersmoother_params.coarse_hypersmoothed_img"),
            get_lines=lambda ilm, rpe: _line_dict(
                hypersmoothed=_get(rpe, "hypersmoother_params.hypersmoother_y_dp"),
            ),
        ),
        ReadmePanel(
            slug="04_flattened_to_hypersmoother",
            title="B-scan flattened to coarse guide",
            description=(
                "The B-scan after warping to the coarse guide. This makes the later search region more stable "
                "across columns and reduces the effect of broad curvature."
            ),
            source_functions=("flattening_utility_functions.flatten_to_path", "ssf.step_rpe_hypersmoother_3_7_26"),
            get_array=lambda ilm, rpe: _get(rpe, "hypersmoothed_img"),
            get_lines=lambda ilm, rpe: _line_dict(ilm_seg_flat=_get(rpe, "ilm_seg_flat")),
        ),
        ReadmePanel(
            slug="05_downsampled_working_image",
            title="Downsampled working image",
            description=(
                "`step_rpe_downsample_and_preprocess` creates a smaller working image for the first RPE pass."
            ),
            source_functions=("ssf.step_rpe_downsample_and_preprocess",),
            get_array=lambda ilm, rpe: _get(rpe, "downsampled_img"),
        ),
        ReadmePanel(
            slug="06_boundary_enhancement",
            title="Filtered axial-gradient image",
            description=(
                "`step_rpe_compute_enhancement2` emphasizes the relevant axial transition near the RPE complex. "
                "This is the filtered gradient image before column-wise peak suppression."
            ),
            source_functions=("ssf.step_rpe_compute_enhancement2", "suf._boundary_enhance"),
            get_array=lambda ilm, rpe: _get(rpe, "enh")
            if _get(rpe, "enh") is not None
            else _get(rpe, "enh_f"),
        ),
        ReadmePanel(
            slug="07_peak_detection_pre_suppression",
            title="Detected peaks before RPE suppression",
            description=(
                "Column-wise peaks are recreated with the same settings used by "
                "`step_rpe_compute_enhancement2` and are overlaid in red on the incoming filtered "
                "gradient image. These are the candidate peaks evaluated by the subsequent suppression step."
            ),
            source_functions=(
                "ssf.step_rpe_compute_enhancement2",
                "suf.peakSuppressor.extract_smoothed_and_peaks",
                "spu.overlay_peaks_on_image",
            ),
            get_array=lambda ilm, rpe: _rpe_peak_overlay(rpe),
        ),
        ReadmePanel(
            slug="08_peak_suppressed_gradient",
            title="Peak-suppressed axial-gradient image",
            description=(
                "The detected peaks are used to attenuate deeper choroidal and scleral signal. "
                "When at least three peaks are present and the third shallowest peak is weaker than "
                "the first two, signal at and below the intervening valley is multiplied by the "
                "configured suppression factor."
            ),
            source_functions=(
                "ssf.step_rpe_compute_enhancement2",
                "suf.peakSuppressor.peak_suppression_pipeline",
                "suf.peakSuppressor.suppress_below_third_peak_valley",
            ),
            get_array=lambda ilm, rpe: _get(rpe, "peak_suppressed"),
        ),
        ReadmePanel(
            slug="09_lowres_dp_cost",
            title="Low-resolution DP cost after ILM suppression",
            description=(
                "`step_rpe_DP_on_enh_2` suppresses the response surrounding the ILM guide, converts "
                "the remaining peak-suppressed response to a cost image, and uses it for the "
                "preliminary RPE dynamic-programming pass."
            ),
            source_functions=(
                "ssf.step_rpe_DP_on_enh_2",
                "suf.apply_gaussian_tube_suppression",
                "suf.run_DP_on_cost_matrix",
            ),
            get_array=lambda ilm, rpe: 1-_get(rpe, "rpe_enh_DP_cost_raw")
            if _get(rpe, "rpe_enh_DP_cost_raw") is not None
            else _get(rpe, "guided_cost_raw"),
        ),
        ReadmePanel(
            slug="10_preliminary_rpe_on_raw",
            title="Preliminary RPE segmentation on the raw B-scan",
            description=(
                "After `step_rpe_upsample` and `step_rpe_unsmooth`, the preliminary RPE estimate is "
                "returned to the original B-scan coordinate space."
            ),
            source_functions=("ssf.step_rpe_upsample", "ssf.step_rpe_unsmooth"),
            get_array=lambda ilm, rpe: _get(rpe, "original_image"),
            get_lines=lambda ilm, rpe: _line_dict(
                # hypersmoothed=_get(rpe, "hypersmoother_params.hypersmoother_path"),
                rpe_smooth=_get(rpe, "rpe_smooth"),
            ),
        ),
        ReadmePanel(
            slug="11_flattened_to_preliminary_rpe",
            title="B-scan flattened to the preliminary RPE",
            description=(
                "`step_rpe_highres_smooth` flattens the original B-scan to the preliminary RPE "
                "segmentation. This more precise flattening supports the subsequent high-resolution "
                "gradient and horizontal-blurring operations."
            ),
            source_functions=(
                "ssf.step_rpe_highres_smooth",
                "flattening_utility_functions.flatten_to_path",
            ),
            get_array=lambda ilm, rpe: _get(rpe, "highres_smoothed_img"),
        ),
        ReadmePanel(
            slug="12_highres_hblur_down",
            title="Horizontally blurred downward-gradient response",
            description=(
                "The high-resolution bright-to-dark axial-gradient response is anisotropically "
                "blurred in the horizontal direction. This reinforces horizontally aligned retinal "
                "bands while smearing less consistently oriented choroidal structures. This is "
                "`hblur_down` from `diff_boundary_enhance_and_blur_horiz`."
            ),
            source_functions=(
                "ssf.step_rpe_highres_diff_enh",
                "suf.diff_boundary_enhance_and_blur_horiz",
            ),
            get_array=lambda ilm, rpe: _highres_hblur_down(rpe),
        ),
        ReadmePanel(
            slug="13_highres_hblur_up",
            title="Horizontally blurred upward-gradient response",
            description=(
                "The high-resolution dark-to-bright axial-gradient response is anisotropically "
                "blurred in the horizontal direction using the corresponding `up_hblur` setting. "
                "This is `hblur_up` from `diff_boundary_enhance_and_blur_horiz`."
            ),
            source_functions=(
                "ssf.step_rpe_highres_diff_enh",
                "suf.diff_boundary_enhance_and_blur_horiz",
            ),
            get_array=lambda ilm, rpe: _highres_hblur_up(rpe),
        ),
        ReadmePanel(
            slug="14_highres_diff_image",
            title="High-resolution differential gradient image",
            description=(
                "The horizontally blurred upward-gradient response is subtracted from the "
                "horizontally blurred downward-gradient response, negative values are removed, "
                "and the result is normalized. This reduces residual choroidal signal while "
                "preserving the horizontally aligned RPE complex."
            ),
            source_functions=(
                "ssf.step_rpe_highres_smooth",
                "ssf.step_rpe_highres_diff_enh",
                "suf.diff_boundary_enhance_and_blur_horiz",
            ),
            get_array=lambda ilm, rpe: _highres_diff_down_up(rpe),
        ),
        ReadmePanel(
            slug="15_lower_edge_of_tubed",
            title="RPE-constrained high-resolution candidate image",
            description=(
                "The preliminary RPE segmentation suppresses regions distant from the expected RPE "
                "location. A final narrow axial-gradient operation sharpens the lower edge used by "
                "the subsequent two-layer DP pathways."
            ),
            source_functions=(
                "ssf.step_rpe_highres_diff_enh",
                "suf.apply_gaussian_tube_mul",
                "suf._normalized_axial_gradient",
            ),
            get_array=lambda ilm, rpe: _get(rpe, "highres_ctx.lower_edge_of_tubed"),
        ),
        ReadmePanel(
            slug="16_original_two_layer_dp",
            title="Standard two-layer DP proposal",
            description=(
                "`step_rpe_highres_DP_two_layer` simultaneously estimates two bright retinal "
                "surfaces within the high-resolution DP image band. Both native band-coordinate "
                "paths are overlaid."
            ),
            source_functions=("ssf.step_rpe_highres_DP_two_layer", "two_surface_utils"),
            get_array=lambda ilm, rpe: _get(rpe, "two_layer_dp_ctx.img_band"),
            get_lines=lambda ilm, rpe: _line_dict(
                original_method_y1=_get(rpe, "two_layer_dp_ctx.y1"),
                original_method_y2=_get(rpe, "two_layer_dp_ctx.y2"),
            ),
        ),
        ReadmePanel(
            slug="17_choroidal_two_layer_dp",
            title="Choroidal-oriented two-layer proposal",
            description=(
                "`step_rpe_highres_DP_two_layer_choroidal` reruns the paired-surface logic using its "
                "choroidal-oriented DP image band."
            ),
            source_functions=("ssf.step_rpe_highres_DP_two_layer_choroidal",),
            get_array=lambda ilm, rpe: _get(rpe, "two_layer_dp_ctx_choroidal.img_band"),
            get_lines=lambda ilm, rpe: _line_dict(
                choroidal_method_y1=_get(rpe, "two_layer_dp_ctx_choroidal.y1"),
                choroidal_method_y2=_get(rpe, "two_layer_dp_ctx_choroidal.y2"),
            ),
        ),
        ReadmePanel(
            slug="18_ez_two_layer_dp",
            title="EZ-oriented two-layer proposal",
            description=(
                "`step_rpe_highres_DP_two_layer_EZ` estimates paired surfaces within its EZ-oriented "
                "high-resolution DP image band."
            ),
            source_functions=("ssf.step_rpe_highres_DP_two_layer_EZ",),
            get_array=lambda ilm, rpe: _get(rpe, "two_layer_dp_ctx_EZ.img_band"),
            get_lines=lambda ilm, rpe: _line_dict(
                EZ_method_y1=_get(rpe, "two_layer_dp_ctx_EZ.y1"),
                EZ_method_y2=_get(rpe, "two_layer_dp_ctx_EZ.y2"),
            ),
        ),
        ReadmePanel(
            slug="19_vertical_shift_refinement",
            title="Vertical-shift refinement",
            description=(
                "`step_rpe_vertical_shift_refine` aligns and refines the proposal lines vertically "
                "before export."
            ),
            source_functions=("ssf.step_rpe_vertical_shift_refine",),
            get_array=lambda ilm, rpe: _get(rpe, "original_image"),
            get_lines=lambda ilm, rpe: _line_dict(
                original_shifted=_get(rpe, "two_layer_dp_ctx.y2_vertical_shifted"),
                choroidal_shifted=_get(rpe, "two_layer_dp_ctx_choroidal.y1_vertical_shifted"),
                EZ_shifted=_get(rpe, "two_layer_dp_ctx_EZ.y2_vertical_shifted"),
            ),
        ),
        ReadmePanel(
            slug="20_final_exported_rpe",
            title=f"Final exported RPE line: {final_rpe_pathway}",
            description=(
                "Final RPE path selected for display from the original, choroidal, or EZ pathway. "
                "Only the selected RPE line is shown on the original B-scan."
            ),
            source_functions=("setup_data/02_segment_ILM_RPE.py::extract_lite",),
            get_array=lambda ilm, rpe: _get(rpe, "original_image"),
            get_lines=lambda ilm, rpe, pathway=final_rpe_pathway: _line_dict(
                rpe_smooth=_selected_final_rpe_line(rpe, pathway),
            ),
        ),
    ]


def write_manifest_md(rows: list[dict], out_path: Path, title: str) -> None:
    lines = [f"# {title}", ""]
    for row in rows:
        rel = Path(row["filename"]).name
        lines.extend(
            [
                f"## {row['index']:02d}. {row['title']}",
                "",
                f'<p align="center"><img src="{rel}" width="90%"></p>',
                "",
                row["description"],
                "",
                "**Code references:** " + ", ".join(f"`{s}`" for s in row["source_functions"]),
                "",
            ]
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def save_segmentation_readme_panels(
    ctx_ilm,
    ctx_rpe,
    *,
    out_dir: str | Path,
    prefix: str = "seg",
    panels: Optional[list[ReadmePanel]] = None,
    dpi: int = 300,
    transparent: bool = True,
    write_manifest: bool = True,
    final_rpe_pathway: str = "original",
) -> list[dict]:
    """Save the available README panels and return manifest rows."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    panels = panels or default_segmentation_readme_panels(
        final_rpe_pathway=final_rpe_pathway,
    )

    rows: list[dict] = []
    for idx, panel in enumerate(panels, start=1):
        arr = panel.get_array(ctx_ilm, ctx_rpe)
        if not _valid_array(arr):
            print(f"Skipping missing/non-image panel: {panel.slug}")
            continue

        filename = f"{prefix}_{idx:02d}_{panel.slug}.png"
        out_path = out_dir / filename

        lines = panel.get_lines(ctx_ilm, ctx_rpe)
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
            out_dir / "panels_manifest.md",
            title="Segmentation algorithm panels",
        )

    return rows


