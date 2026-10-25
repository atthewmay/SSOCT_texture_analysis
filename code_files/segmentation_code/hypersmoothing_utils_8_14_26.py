#Reviewed
"""
Nasal-to-ONH hypersmoother.

Annotation labels:
    0 = background
    1 = ONH
    2 = fovea

Only label 1 is treated as ONH.

New behavior:
    DP runs normally. Before smooth_rpe_line(), the raw DP path across the
    ONH is replaced by a straight line connecting the immediately adjacent
    RPE points on the left and right sides of the ONH.
"""

from dataclasses import dataclass, replace
from pathlib import Path
from textwrap import fill
from unittest.mock import patch

import cv2
import numpy as np
from joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter

import code_files.segmentation_code.flattening_utility_functions as fuf
import code_files.segmentation_code.segmentation_plot_utils as spu
import code_files.segmentation_code.segmentation_utility_functions as suf
from code_files.segmentation_code.custom_dataclasses import HyperPreprocessParams, ImageBundle


OUTPUT_ROOT = Path("/Volumes/T9/iowa_research/Han_AIR_Dec_2025/results")
ONH_LABEL = 1


@dataclass(frozen=True)
class NasalHypersmootherParams:
    downsample_factor: int = 4
    kernel_size: int = 20
    grad_weight: float = 0.6
    blur_sigma: int = 4  # must stay int; existing code passes it to cv2.blur
    suppression_sigma: float = 12.0
    peak_prominence: float = 0.02
    rigidity: float = 40.0
    lambda_step_in_onh_region: float = 0.0001
    nasal_gradient_blur_sigma: float = 1.5
    nasal_suppression_rolloff_sigma: float = 16.0
    nasal_max_suppression: float = 1.0

    def __post_init__(self):
        if not isinstance(self.downsample_factor, (int, np.integer)):
            raise TypeError("downsample_factor must be an integer")
        if not isinstance(self.kernel_size, (int, np.integer)):
            raise TypeError("kernel_size must be an integer")
        if not isinstance(self.blur_sigma, (int, np.integer)):
            raise TypeError("blur_sigma must be an integer because the existing hypersmoother passes it to cv2.blur")
        if self.downsample_factor <= 0 or self.kernel_size <= 0 or self.blur_sigma <= 0:
            raise ValueError("downsample_factor, kernel_size, and blur_sigma must be positive")
        if not 0 <= self.grad_weight <= 1:
            raise ValueError("grad_weight must be between 0 and 1")
        if not 0 <= self.nasal_max_suppression <= 1:
            raise ValueError("nasal_max_suppression must be between 0 and 1")


GOLDEN_PARAMS = NasalHypersmootherParams()


def _get_onh_only_mask(ONH_info):
    """Label 1 = ONH; label 2 (fovea) is ignored."""
    return None if ONH_info is None else np.asarray(ONH_info) == ONH_LABEL


def _get_onh_bounds(ONH_info, target_width: int):
    """Return inclusive ONH column bounds in target-width coordinates."""
    onh = _get_onh_only_mask(ONH_info)
    if onh is None:
        return None
    cols_mask = onh if onh.ndim == 1 else np.any(onh, axis=tuple(range(onh.ndim - 1)))
    if not cols_mask.any():
        return None

    cols = np.flatnonzero(cols_mask)
    scale = target_width / cols_mask.size
    left = int(np.clip(np.floor(cols[0] * scale), 0, target_width - 1))
    right = int(np.clip(np.ceil((cols[-1] + 1) * scale) - 1, 0, target_width - 1))
    return left, right


def _get_nasal_columns(ONH_info, target_width: int):
    """Return columns nasal to ONH; no label-1 ONH means no modification."""
    bounds = _get_onh_bounds(ONH_info, target_width)
    if bounds is None:
        return None

    left, right = bounds
    nasal = np.zeros(target_width, dtype=bool)
    if 0.5 * (left + right) < target_width / 2:
        nasal[:left] = True
    else:
        nasal[right + 1:] = True
    return nasal


def _linearize_path_across_onh(y_path, ONH_info):
    """
    Replace only the ONH portion of a path with a straight line connecting
    the immediately adjacent left and right non-ONH path points.
    """
    y = np.asarray(y_path).copy()
    bounds = _get_onh_bounds(ONH_info, len(y))
    if bounds is None:
        return y

    left, right = bounds
    x0, x1 = left - 1, right + 1
    if x0 < 0 or x1 >= len(y):
        return y  # cannot bridge if ONH touches an image edge

    x = np.arange(left, right + 1)
    y[left:right + 1] = np.interp(x, [x0, x1], [y[x0], y[x1]])
    return y


def _make_preprocess_params(p: NasalHypersmootherParams, ilm_line=None):
    return HyperPreprocessParams(
        ILM_line=ilm_line,
        grad_weight=float(p.grad_weight),
        kernel_size=int(p.kernel_size),
        blur_sigma=int(p.blur_sigma),
        suppression_sigma=float(p.suppression_sigma),
        peak_prominence=float(p.peak_prominence),
    )


def _make_bundle(img, preprocess_params):
    return suf.HypersmoothPreprocessors._big_blur_grad_combined_peak_suppressed_bundle(img, preprocess_params)


def _replace_nasal_gradient_fields(baseline: ImageBundle, candidate: ImageBundle, nasal_cols):
    if nasal_cols is None:
        return baseline
    for name in ("grad_raw", "grad_ilm_suppressed", "grad_shift", "grad_peak_suppressed", "peak_img"):
        a, b = getattr(baseline, name, None), getattr(candidate, name, None)
        if a is None or b is None:
            continue
        modified = a.copy()
        modified[:, nasal_cols] = b[:, nasal_cols]
        setattr(baseline, name, modified)
    return baseline


def _suppress_blurred_nasal_to_onh(bundle, ilm_line, nasal_cols, *, rolloff_sigma, max_suppression):
    if nasal_cols is None:
        return bundle, None

    height, width = bundle.blurred.shape
    if len(ilm_line) != width:
        raise ValueError(f"ILM/bundle width mismatch: {len(ilm_line)} versus {width}")

    weights = np.zeros((height, width), dtype=np.float32)
    rows = np.arange(height, dtype=np.float32)
    for x in np.flatnonzero(nasal_cols):
        distance = rows - float(ilm_line[x])
        weights[:, x] = float(max_suppression) * np.exp(-0.5 * (distance / float(rolloff_sigma)) ** 2)

    bundle.blurred = bundle.blurred.copy()
    bundle.blurred *= 1.0 - weights
    return bundle, weights


def _recombine_nasal_only(bundle, original_combined, nasal_cols, *, grad_weight):
    if nasal_cols is None:
        bundle.combined_peak_suppressed = original_combined.copy()
        bundle.output = bundle.combined_peak_suppressed
        return bundle

    recombined = suf.normalize_image(
        bundle.grad_peak_suppressed * float(grad_weight) + bundle.blurred * (1.0 - float(grad_weight)),
        zero_min=True,
    )
    combined = original_combined.copy()
    combined[:, nasal_cols] = recombined[:, nasal_cols]
    bundle.combined_peak_suppressed = bundle.output = combined
    return bundle


def _run_one_nasal_hypersmoother_config(ctx, p: NasalHypersmootherParams):
    ds = int(p.downsample_factor)
    raw_ds = cv2.resize(ctx.img.copy(), (0, 0), fx=1.0 / ds, fy=1.0 / ds)
    baseline_img = gaussian_filter(raw_ds, sigma=int(p.blur_sigma))
    ilm_ds = ctx.ilm_ctx.ilm_smooth[::ds] / ds
    preprocess_params = _make_preprocess_params(p, ilm_ds)

    nasal_cols = _get_nasal_columns(ctx.ONH_region, baseline_img.shape[1])
    onh_only_mask = _get_onh_only_mask(ctx.ONH_region)

    # Baseline bundle.
    bundle = _make_bundle(baseline_img, preprocess_params)
    original_combined = bundle.combined_peak_suppressed.copy()

    # Higher-resolution gradient input nasal to ONH.
    nasal_gradient_source = gaussian_filter(raw_ds, sigma=float(p.nasal_gradient_blur_sigma))
    composite_gradient_input = baseline_img.copy()
    if nasal_cols is not None:
        composite_gradient_input[:, nasal_cols] = nasal_gradient_source[:, nasal_cols]

    candidate = _make_bundle(composite_gradient_input, preprocess_params)
    bundle = _replace_nasal_gradient_fields(bundle, candidate, nasal_cols)

    # Suppress blurred branch around ILM nasal to ONH.
    blurred_pre_suppression = bundle.blurred.copy()
    bundle, suppression_weights = _suppress_blurred_nasal_to_onh(
        bundle, ilm_ds, nasal_cols,
        rolloff_sigma=p.nasal_suppression_rolloff_sigma,
        max_suppression=p.nasal_max_suppression,
    )
    blurred_post_suppression = bundle.blurred.copy()
    bundle = _recombine_nasal_only(bundle, original_combined, nasal_cols, grad_weight=p.grad_weight)

    # DP runs normally. Intercept only the smoothing step:
    # raw DP -> straight line through ONH -> existing smooth_rpe_line().
    original_smoother = suf.smooth_rpe_line

    def smooth_with_onh_bridge(y_dp, rigidity=10):
        return original_smoother(_linearize_path_across_onh(y_dp, onh_only_mask), rigidity=rigidity)

    with patch.object(suf, "smooth_rpe_line", smooth_with_onh_bridge):
        coarse, raw_y_dp, y_dp, cost = suf.rpe_hypersmoother_DP_3_7_26(
            coarse=bundle.combined_peak_suppressed,
            rigidity=float(p.rigidity),
            ONH_info=onh_only_mask,
            lambda_step_in_ONH_region=float(p.lambda_step_in_onh_region),
        )

    linearized_y_dp = _linearize_path_across_onh(raw_y_dp, onh_only_mask)
    raw_dp_path = suf.upsample_path(raw_y_dp, vertical_factor=ds, horizontal_factor=ds)
    linearized_path = suf.upsample_path(linearized_y_dp, vertical_factor=ds, horizontal_factor=ds)
    hypersmoother_path = suf.upsample_path(y_dp, vertical_factor=ds, horizontal_factor=ds)
    ilm_full = suf.upsample_path(ilm_ds, vertical_factor=ds, horizontal_factor=ds)

    return dict(
        run_params=p,
        composite_gradient_input=composite_gradient_input,
        blurred_pre_suppression=blurred_pre_suppression,
        blurred_post_suppression=blurred_post_suppression,
        suppression_weights=suppression_weights,
        bundle=bundle,
        coarse=coarse,
        raw_y_dp=raw_y_dp,
        linearized_y_dp=linearized_y_dp,
        y_dp=y_dp,
        cost=cost,
        raw_dp_path=raw_dp_path,
        linearized_path=linearized_path,
        hypersmoother_path=hypersmoother_path,
        ilm_full=ilm_full,
    )


def step_rpe_hypersmoother_8_5_26(ctx):
    result = _run_one_nasal_hypersmoother_config(ctx, GOLDEN_PARAMS)
    hp = ctx.hypersmoother_params
    hp.coarse_hypersmoothed_img = result["coarse"]
    hp.hypersmoother_y_dp = result["y_dp"]
    hp.hypersmoother_path = result["hypersmoother_path"]

    ctx.hypersmoothed_img, hp.hypersmoother_shift_y_full, hp.hypersmoother_target_y = fuf.flatten_to_path(
        ctx.img, hp.hypersmoother_path
    )
    ctx.ilm_seg_flat = fuf.warp_line_by_shift(ctx.ilm_seg, hp.hypersmoother_shift_y_full, direction="to_flat")
    ctx.ilm_seg = ctx.ilm_seg_flat
    ctx.img = ctx.hypersmoothed_img
    return ctx


def step_rpe_hypersmoother_nasal_param_sweep(ctx):
    sweep_params = [
        replace(GOLDEN_PARAMS, nasal_gradient_blur_sigma=grad, nasal_suppression_rolloff_sigma=rolloff)
        for grad in (1.5, 2.5)
        for rolloff in (16.0, 32.0)
    ]
    results = Parallel(n_jobs=min(8, len(sweep_params)))(
        delayed(_run_one_nasal_hypersmoother_config)(ctx, p) for p in sweep_params
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    safe_id = str(ctx.ID).replace("/", "_")
    AB = spu.ArrayBoard(
        skip=False,
        plt_display=False,
        temp_fig_location=OUTPUT_ROOT / f"hypersmoother_ONH_nasal_param_sweep_{safe_id}.png",
    )

    for result in results:
        p, bundle = result["run_params"], result["bundle"]
        suffix = f"grad blur={p.nasal_gradient_blur_sigma}, rolloff={p.nasal_suppression_rolloff_sigma}"

        AB.add(result["composite_gradient_input"], title=fill(f"gradient input {suffix}", 30))
        AB.add(bundle.grad_raw, title=fill(f"gradient raw {suffix}", 30))
        AB.add(bundle.grad_peak_suppressed, title=fill(f"gradient peak suppressed {suffix}", 30))
        AB.add(result["blurred_pre_suppression"], title=fill(f"blurred PRE suppression {suffix}", 30))
        AB.add(result["blurred_post_suppression"], title=fill(f"blurred POST suppression {suffix}", 30))
        if result["suppression_weights"] is not None:
            AB.add(result["suppression_weights"], title=fill(f"suppression weights {suffix}", 30))
        AB.add(bundle.combined_peak_suppressed, title=fill(f"combined cost input {suffix}", 30))
        AB.add(result["cost"], title=fill(f"DP cost {suffix}", 30))
        AB.add(ctx.img, lines={"raw DP": result["raw_dp_path"], "ilm": result["ilm_full"]}, title=fill(f"raw DP {suffix}", 30))
        AB.add(ctx.img, lines={"linearized DP": result["linearized_path"], "ilm": result["ilm_full"]}, title=fill(f"ONH bridged DP {suffix}", 30))
        AB.add(ctx.img, lines={"hypersmoother": result["hypersmoother_path"], "ilm": result["ilm_full"]}, title=fill(f"smoothed path {suffix}", 30))

    AB.render()
    raise Exception("done plotting nasal hypersmoother parameter sweep")

