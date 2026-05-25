from __future__ import annotations
#Reviewed

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

    # Figure aspect matches image aspect; this avoids unnecessary padding.
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

        style = spu.LAYER_STYLE.get(name, None)
        x = np.arange(line.shape[0])
        if style is not None:
            ax.plot(x, line, style.get("fmt", "-"), lw=line_width, alpha=style.get("alpha", 0.9))
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


def default_segmentation_readme_panels() -> list[ReadmePanel]:
    """Panels matching the current unified RPE pipeline narrative."""
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
            title="Coarse hypersmoother image",
            description=(
                "The low-resolution image/cost surface used for the coarse guide. "
                "This is useful for explaining why the guide prefers the broad RPE/choroid complex."
            ),
            source_functions=("ssf.step_rpe_hypersmoother_3_7_26",),
            get_array=lambda ilm, rpe: _get(rpe, "hypersmoother_params.coarse_hypersmoothed_img"),
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
            title="Boundary enhancement",
            description=(
                "`step_rpe_compute_enhancement2` builds an image that emphasizes the relevant axial transition "
                "near the RPE complex. Computes axial graident, blurs, and then suppresses peaks below the top (most anteriorly oriented) "
                "2 peaks to reduce choroidal signal."
            ),
            source_functions=("ssf.step_rpe_compute_enhancement2","suf.peakSuppressor"),
            get_array=lambda ilm, rpe: _get(rpe, "enh_f") if _get(rpe, "enh_f") is not None else _get(rpe, "enh"),
        ),
        ReadmePanel(
            slug="07_lowres_dp_cost",
            title="Low-resolution DP cost",
            description=(
                "`step_rpe_DP_on_enh_2` runs a globally optimized dynamic-programming path through the enhanced image. Cost is increased near ILM line."
            ),
            source_functions=("ssf.step_rpe_DP_on_enh_2",),
            get_array=lambda ilm, rpe: _get(rpe, "rpe_enh_DP_cost_raw")
            if _get(rpe, "rpe_enh_DP_cost_raw") is not None
            else _get(rpe, "guided_cost_raw"),
            # get_lines=lambda ilm, rpe: _line_dict(rpe_smooth=_get(rpe, "rpe_smooth")),
        ),
        ReadmePanel(
            slug="08_lowres_rpe_on_raw",
            title="Low-resolution RPE path on raw image",
            description=(
                "After `step_rpe_upsample` and `step_rpe_unsmooth`, the coarse RPE estimate is returned "
                "to the original B-scan coordinate space."
            ),
            source_functions=("ssf.step_rpe_upsample", "ssf.step_rpe_unsmooth"),
            get_array=lambda ilm, rpe: _get(rpe, "original_image"),
            get_lines=lambda ilm, rpe: _line_dict(
                hypersmoothed=_get(rpe, "hypersmoother_params.hypersmoother_path"),
                rpe_smooth=_get(rpe, "rpe_smooth"),
            ),
        ),
        ReadmePanel(
            slug="09_highres_diff_image",
            title="High-resolution differential image",
            description=(
                "`step_rpe_highres_diff_enh` builds a local high-resolution differential/gradient image "
                "around the RPE complex."
            ),
            source_functions=("ssf.step_rpe_highres_smooth", "ssf.step_rpe_highres_diff_enh"),
            get_array=lambda ilm, rpe: _get(rpe, "highres_ctx.diff_down_up"),
        ),
        ReadmePanel(
            slug="10_lower_edge_of_tubed",
            title="Lower-edge candidate image",
            description=(
                "A high-resolution image with regions far from RPE proposal suppressed, highlighting the lower edge used by later DP refinement."
            ),
            source_functions=("ssf.step_rpe_highres_diff_enh",),
            get_array=lambda ilm, rpe: _get(rpe, "highres_ctx.lower_edge_of_tubed"),
        ),
        ReadmePanel(
            slug="11_original_two_layer_dp",
            title="Original two-layer DP proposal",
            description=(
                "`step_rpe_highres_DP_two_layer` estimates a paired-surface proposal in the high-resolution band."
            ),
            source_functions=("ssf.step_rpe_highres_DP_two_layer", "two_surface_utils"),
            get_array=lambda ilm, rpe: _get(rpe, "original_image"),
            get_lines=lambda ilm, rpe: _line_dict(
                original_method_y1=_get(rpe, "two_layer_dp_ctx.y1_rescaled"),
                original_method_y2=_get(rpe, "two_layer_dp_ctx.y2_rescaled"),
            ),
        ),
        ReadmePanel(
            slug="12_choroidal_two_layer_dp",
            title="Choroidal-oriented proposal",
            description=(
                "`step_rpe_highres_DP_two_layer_choroidal` reruns/refines the RPE paired-surface logic with a "
                "focus on images with high choroidal signal."
            ),
            source_functions=("ssf.step_rpe_highres_DP_two_layer_choroidal",),
            get_array=lambda ilm, rpe: _get(rpe, "original_image"),
            get_lines=lambda ilm, rpe: _line_dict(
                choroidal_method_y1=_get(rpe, "two_layer_dp_ctx_choroidal.y1_rescaled"),
                choroidal_method_y2=_get(rpe, "two_layer_dp_ctx_choroidal.y2_rescaled"),
            ),
        ),
        ReadmePanel(
            slug="13_ez_two_layer_dp",
            title="EZ-oriented proposal",
            description=(
                "`step_rpe_highres_DP_two_layer_EZ` preserves a paired-surface RPE proposal for "
                "images with robust EZ band."
            ),
            source_functions=("ssf.step_rpe_highres_DP_two_layer_EZ",),
            get_array=lambda ilm, rpe: _get(rpe, "original_image"),
            get_lines=lambda ilm, rpe: _line_dict(
                EZ_method_y1=_get(rpe, "two_layer_dp_ctx_EZ.y1_rescaled"),
                EZ_method_y2=_get(rpe, "two_layer_dp_ctx_EZ.y2_rescaled"),
            ),
        ),
        ReadmePanel(
            slug="14_vertical_shift_refinement",
            title="Vertical-shift refinement",
            description=(
                "`step_rpe_vertical_shift_refine` aligns/refines the proposal lines vertically before export."
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
            slug="15_final_exported_lines",
            title="Final exported lines",
            description=(
                "Final compact set of paths saved into the segmentation `.npz` outputs for downstream flattening "
                "and texture projection."
            ),
            source_functions=("setup_data/02_segment_ILM_RPE.py::extract_lite",),
            get_array=lambda ilm, rpe: _get(rpe, "original_image"),
            get_lines=lambda ilm, rpe: _line_dict(
                ilm_smooth=_get(ilm, "ilm_smooth"),
                rpe_smooth=_get(rpe, "rpe_smooth"),
                original_RPE=_get(rpe, "two_layer_dp_ctx.y2_vertical_shifted")
                if _get(rpe, "two_layer_dp_ctx.y2_vertical_shifted") is not None
                else _get(rpe, "two_layer_dp_ctx.y2_rescaled"),
                choroidal_RPE=_get(rpe, "two_layer_dp_ctx_choroidal.y1_vertical_shifted")
                if _get(rpe, "two_layer_dp_ctx_choroidal.y1_vertical_shifted") is not None
                else _get(rpe, "two_layer_dp_ctx_choroidal.y1_rescaled"),
                EZ_RPE=_get(rpe, "two_layer_dp_ctx_EZ.y2_vertical_shifted")
                if _get(rpe, "two_layer_dp_ctx_EZ.y2_vertical_shifted") is not None
                else _get(rpe, "two_layer_dp_ctx_EZ.y2_rescaled"),
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
) -> list[dict]:
    """Save the available README panels and return manifest rows."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    panels = panels or default_segmentation_readme_panels()

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


def step_rpe_save_readme_panels(ctx):
    """Optional RPE pipeline step: save README panels at the end of a debug run.

    Add this as the final step in a one-off/debug pipeline:

        RPE_STEPS_readme = sp.RPE_STEPS_unified_3_19_26 + [
            step_rpe_save_readme_panels,
        ]

    It expects `ctx.ilm_ctx` to be present, as in `sp.process_bscan_1_3_26`.
    """
    # out_dir = Path("docs/assets/segmentation_panels") / str(ctx.ID)
    out_dir = Path("docs/assets/segmentation_panels") / "demo_panels"
    save_segmentation_readme_panels(
        ctx_ilm=ctx.ilm_ctx,
        ctx_rpe=ctx,
        out_dir=out_dir,
        # prefix=str(ctx.ID).replace("/", "_").replace(":", "_"),
        prefix="demo_segmentations",
    )
    return ctx
