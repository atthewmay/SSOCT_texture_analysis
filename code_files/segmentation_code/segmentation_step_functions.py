
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # adds Han_AIR/ to path
import code_files.segmentation_code.segmentation_utility_functions as suf
import code_files.segmentation_code.segmentation_plot_utils as spu
# import code_files.segmentation_utils as su
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Callable, List
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

# type alias for step functions
RPEStepFn = Callable[["RPEContext"], "RPEContext"]
ILMStepFn = Callable[["ILMContext"], "ILMContext"]


# -----------------------------
#  RPE config / context
# -----------------------------

@dataclass(frozen=True)
class RPEConfig:
    # “version-defining” hyperparameters
    seed_every: int = 3
    neighbourhood: int = 2
    rigidity: float = 30.0

    # enhancement / kernel knobs
    vks_false_factor: float = 0.75
    vks_true_factor: float = 1.33
    k_rows_factor: float = 1.0
    k_cols_factor: float = 1.0

    blur_ksize: int = 25

    # hysteresis on prob
    hysteresis_high: float = 0.001
    hysteresis_low: float = 0.0001

    # guided DP knobs
    lambda_step_guided: float = 0.1
    alpha_guide: float = 1.0
    sigma_guide: float = 1.0
    ilm_margin_divisor: float = 3.0
    ilm_max_factor: float = 2.0
    ONH_value_factor: float = 0.5

    # tube smoother
    tube_lambda_step: float = 0.1
    tube_sigma_guide: float = 20.0
    tube_max_step: int = 3

    # downsampling / upsampling
    downsample_factor: float = 1.5
    original_height: int = 512


@dataclass
class RPEContext:
    # inputs
    idx: int
    original_image: np.ndarray
    ONH_region: Any
    cfg: RPEConfig
    img: np.ndarray = None              # working image (may be downsampled)
    ilm_seg: np.ndarray = None
    kwargs: dict = field(default_factory=dict)

    # runtime config-ish
    d_vertical: float = 1.0
    d_horizontal: float = 1.0

    # intermediates
    images: Dict[str, np.ndarray] = field(default_factory=dict)
    enh: Optional[np.ndarray] = None
    enh_f: Optional[np.ndarray] = None
    peak_suppressed: Optional[np.ndarray] = None
    peak_suppressed_recalculated: Optional[np.ndarray] = None
    seeds: Optional[np.ndarray] = None
    prob: Optional[np.ndarray] = None
    edge: Optional[np.ndarray] = None

    rpe_raw: Optional[np.ndarray] = None
    ilm_margin: Optional[float] = None

    rpe_guided: Optional[np.ndarray] = None
    guided_cost: Optional[np.ndarray] = None
    guided_cost_raw: Optional[np.ndarray] = None

    rpe_guided_tube_smoothed: Optional[np.ndarray] = None
    guided_cost_tube_smoothed: Optional[np.ndarray] = None
    guided_cost_raw_tube_smoothed: Optional[np.ndarray] = None

    rpe_smooth: Optional[np.ndarray] = None

    # debug / history
    debug: bool = True
    history: Dict[str, Any] = field(default_factory=dict)
    # _initialized: bool = field(default=False, init=False, repr=False)
    def log_history(self,name,attribute):
        """logs a history, probably prior to chnaging it, only occurs if debug is true"""
        if not self.debug:
            return
        self.history[name] = attribute

    def __getattr__(self, name: str):
        """
        Only called *if normal attribute lookup fails*.
        Try history[name] as a fallback.
        """
        history = object.__getattribute__(self, "history")
        if name in history:
            return history[name]
        raise AttributeError(f"{type(self).__name__!s} object has no attribute {name!r}")

    # def __getattribute__(self, name: str):
    #     """
    #     First try normal attribute lookup.
    #     If that fails, look in `self.history` and return a stored snapshot.
    #     If still not found, raise AttributeError as usual.
    #     """
    #     try:
    #         # Normal attribute resolution (including dataclass fields)
    #         return object.__getattribute__(self, name)
    #     except AttributeError:
    #         # Fallback to history dict if present
    #         history = object.__getattribute__(self, "history")
    #         if name in history:
    #             return history[name]
    #         # Nothing found → re-raise the normal error
    #         raise

    # def __post_init__(self):
    #     # mark that init is done; future setattr calls will be logged
    #     object.__setattr__(self, "_initialized", True)

    # def __setattr__(self, name: str, value: Any) -> None:
    #     # actually set the attribute
    #     object.__setattr__(self, name, value)
    #     # only log once initialized
    #     if getattr(self, "_initialized", False):
    #         self.log_history(name, value)

    # def log_history(self, name: str, value: Any) -> None: # Again these are fairly useless unless we become memory constrained and delete attributes when in production
    #     if not self.debug:
    #         return
    #     if name in {"history", "debug", "_initialized", "cfg", "kwargs"}:
    #         return
    #     if name.startswith("_"):
    #         return
    #     self.history[name] = value


# -----------------------------
#  ILM config / context
# -----------------------------

@dataclass(frozen=True)
class ILMConfig:
    # downsampling factors
    horizontal_factor: float = 1.5
    vertical_factor: float = 1.5

    # enhancement / kernel knobs
    vks_false_factor: float = 0.75
    vks_true_factor: float = 1.33
    k_rows_factor: float = 1.0
    k_cols_factor: float = 1.0
    blur_ksize: int = 25

    # seed detection
    seeds_radius: int = 15
    seeds_thresh: float = 0.08
    seeds_value_filter: int = 2

    # hysteresis on seeds
    hysteresis_high: float = 0.6
    hysteresis_low: float = 0.15

    # tube smoother for ILM
    tube_sigma_guide: float = 20.0
    tube_lambda_step: float = 0.0
    tube_max_step: int = 5

    original_height: int = 512


@dataclass
class ILMContext:
    idx: int
    original_image: np.ndarray
    cfg: ILMConfig
    img: np.ndarray = None
    ONH_region: Any = None
    kwargs: dict = field(default_factory=dict)

    # downsampling info
    d_vertical: float = 1.0
    d_horizontal: float = 1.0

    # intermediates
    enh: Optional[np.ndarray] = None
    seeds: Optional[np.ndarray] = None
    edge_raw: Optional[np.ndarray] = None
    edge: Optional[np.ndarray] = None

    ilm_raw: Optional[np.ndarray] = None
    ilm_smooth: Optional[np.ndarray] = None
    ilm_tube_cost_raw: Optional[np.ndarray] = None
    ilm_tube_cost_DP_path: Optional[np.ndarray] = None

    # # debug / history
    # debug: bool = False
    # history: Dict[str, Any] = field(default_factory=dict)
    # _initialized: bool = field(default=False, init=False, repr=False)

    # def __post_init__(self):
    #     object.__setattr__(self, "_initialized", True)

    # def __setattr__(self, name: str, value: Any) -> None:
    #     object.__setattr__(self, name, value)
    #     if getattr(self, "_initialized", False):
    #         self.log_history(name, value)

    # def log_history(self, name: str, value: Any) -> None:
    #     if not self.debug:
    #         return
    #     if name in {"history", "debug", "_initialized", "cfg", "kwargs"}:
    #         return
    #     if name.startswith("_"):
    #         return
    #     self.history[name] = value


# -----------------------------
#  Generic pipeline runner
# -----------------------------

def run_pipeline(ctx, steps: List[Callable[[Any], Any]]):
    for step in steps:
        ctx = step(ctx)
    return ctx



# -----------------------------
#  RPE step_ functions
# -----------------------------

def step_rpe_downsample_and_preprocess(ctx: RPEContext) -> RPEContext:
    ctx.img = ctx.original_image.copy()
    # print(f"the shape of img coming in is {ctx.img.shape}")

    d_vertical = ctx.cfg.downsample_factor
    ctx.d_vertical = float(d_vertical)
    ctx.d_horizontal = 1.0

    if ctx.d_vertical != 1.0:
        ctx.img = cv2.resize(
            ctx.img,
            (0, 0),
            fx=1.0 / ctx.d_horizontal,
            fy=1.0 / ctx.d_vertical,
        )

    # print(f"the shape of img after resize is {ctx.img.shape}")
    ctx.img = ctx.img.astype(np.float32)
    ctx.img = gaussian_filter(ctx.img, sigma=5)

    return ctx


def step_rpe_compute_enhancement(ctx: RPEContext) -> RPEContext:
    d_vertical = ctx.d_vertical if ctx.d_vertical != 0 else 1.0

    vks_false = 2 * round((32 / d_vertical) * ctx.cfg.vks_false_factor)
    vks_true  = 2 * round((128 / d_vertical) * ctx.cfg.vks_true_factor)
    k_rows    = 2 * round((20 / d_vertical) * ctx.cfg.k_rows_factor)
    k_cols    = 2 * round(30 * ctx.cfg.k_cols_factor)

    images = suf.compute_enh_diff(
        ctx.img,
        blur_ksize=ctx.cfg.blur_ksize,
        vks_false=vks_false,
        vks_true=vks_true,
        k_rows=k_rows,
        k_cols=k_cols,
    )
    ctx.images = images
    ctx.enh = images["diff"]
    ctx.enh_f = images["enh_f"]
    return ctx


def step_rpe_peak_suppression(ctx: RPEContext) -> RPEContext:
    ctx.peak_suppressed = suf.peakSuppressor.peak_suppression_pipeline(
        ctx.enh,
        ctx.enh,
        ctx.ilm_seg,
        suppression_factor=0,
        third_peak_margin=30,
    )
    return ctx


def step_rpe_seed_selection(ctx: RPEContext) -> RPEContext:
    seeds = suf._nms_columnwise(
        ctx.peak_suppressed, # These params live here for now, but if I tune them later, they should move up into config
        radius=25,
        thresh=0.08,
        vertical_filter=2,
        keeptop=False,
    )

    mask = np.zeros_like(seeds, dtype=bool)
    mask[:, ::ctx.cfg.seed_every] = True
    seeds &= mask

    ctx.seeds = seeds
    return ctx

def step_rpe_recalculate_single_seeded_and_reseed(ctx:RPEContext) -> RPEContext:
    """ a new function that calculates dense seeds, then recalculates the peak_suppressed imge, and then calculates the typical looser seeds.
    This can be modified perhaps by insead of implementing the full recalulation with imputation of enh_f, maybe a softer version of the suppression
    """

    seeds = suf._nms_columnwise(
        ctx.peak_suppressed, # These params live here for now, but if I tune them later, they should move up into config
        radius=25,
        thresh=0.08,
        vertical_filter=2,
        keeptop=False,
    )

    mask = np.zeros_like(seeds, dtype=bool)
    mask[:, :] = True
    seeds &= mask

    ctx.seeds = seeds

    ctx.log_history('original_peak_suppressed', ctx.peak_suppressed.copy())
    pre_suppressed_recalculated = suf.recalculate_single_seeded_cols(ctx.seeds,ctx.peak_suppressed,ctx.enh_f)
    ctx.log_history('pre_suppressed_recalculated',pre_suppressed_recalculated)
    ctx.peak_suppressed = suf.peakSuppressor.peak_suppression_pipeline(
        pre_suppressed_recalculated,
        pre_suppressed_recalculated,
        ctx.ilm_seg,
        suppression_factor=0,
        third_peak_margin=30,
    )

    # ctx.peak_suppressed = suf.peakSuppressor(ctx.seeds,ctx.peak_suppressed,ctx.enh_f)

    seeds = suf._nms_columnwise(
        ctx.peak_suppressed, # These params live here for now, but if I tune them later, they should move up into config
        radius=25,
        thresh=0.08,
        vertical_filter=2,
        keeptop=False,
    )

    mask = np.zeros_like(seeds, dtype=bool)
    mask[:, ::ctx.cfg.seed_every] = True
    seeds &= mask

    ctx.log_history('original_seeds' , ctx.seeds.copy())
    ctx.seeds = seeds
    return ctx


def step_rpe_paths_prob_edge(ctx: RPEContext) -> RPEContext:
    paths = suf._trace_paths(
        ctx.peak_suppressed,
        ctx.seeds,
        ctx.cfg.neighbourhood,
    )
    prob = suf._probability_image(paths, ctx.img.shape)

    # multiply prob by intensity (your MULT_PROB_TIMES_INTENSITY block)
    prob *= ctx.peak_suppressed
    prob /= (prob.max() + 1e-6)

    edge = suf._hysteresis(
        prob,
        high=ctx.cfg.hysteresis_high,
        low=ctx.cfg.hysteresis_low,
    )

    ctx.prob = prob
    ctx.edge = edge
    return ctx


def step_rpe_extract_rpe_raw_and_margin(ctx: RPEContext) -> RPEContext:
    rpe_raw = suf._extract_topbottom_line(
        ctx.edge,
        skip_extreme=0,
        direction="bottom",
    )
    ctx.rpe_raw = rpe_raw

    ctx.ilm_margin = np.nanmedian(rpe_raw - ctx.ilm_seg) // ctx.cfg.ilm_margin_divisor
    return ctx


def step_rpe_guided_dp(ctx: RPEContext) -> RPEContext:
    rpe_guided, guided_cost, guided_cost_raw = suf.guided_dp_rpe(
        img_or_prob=ctx.peak_suppressed,
        ilm_y=ctx.ilm_seg,
        guide_y=ctx.rpe_raw,
        lambda_step=ctx.cfg.lambda_step_guided,
        alpha_guide=ctx.cfg.alpha_guide,
        sigma_guide=ctx.cfg.sigma_guide,
        ilm_margin=int(ctx.ilm_margin),
        ilm_max_factor=ctx.cfg.ilm_max_factor,
        ONH_region=ctx.ONH_region,
        ONH_value_factor=ctx.cfg.ONH_value_factor,
    )

    ctx.rpe_guided = rpe_guided
    ctx.guided_cost = spu.overlay_helper(ctx.ilm_seg, guided_cost)
    ctx.guided_cost_raw = spu.overlay_helper(ctx.ilm_seg, guided_cost_raw)
    return ctx


def step_rpe_tube_smoother(ctx: RPEContext) -> RPEContext:
    (
        rpe_guided_tube_smoothed,
        guided_cost_tube_smoothed,
        guided_cost_raw_tube_smoothed,
    ) = suf.tube_smoother_DP(
        img=ctx.img,
        guide_y=ctx.rpe_guided,
        lambda_step=ctx.cfg.tube_lambda_step,
        max_step=ctx.cfg.tube_max_step,
        sigma_guide=ctx.cfg.tube_sigma_guide,
    )

    ctx.rpe_guided_tube_smoothed = rpe_guided_tube_smoothed
    ctx.guided_cost_tube_smoothed = guided_cost_tube_smoothed
    ctx.guided_cost_raw_tube_smoothed = guided_cost_raw_tube_smoothed

    # spu.save_image_exploration(ctx.img, rpe_guided_tube_smoothed, pickle_save=False) # won't actually run
    return ctx


def step_rpe_smooth_and_upsample(ctx: RPEContext) -> RPEContext:
    rpe_smooth = suf.smooth_rpe_line(
        ctx.rpe_guided_tube_smoothed,
        rigidity=ctx.cfg.rigidity,
    )

    lines = [
        ctx.rpe_raw,
        ctx.rpe_guided,
        ctx.rpe_guided_tube_smoothed,
        rpe_smooth,
    ]
    upsampled = [
        suf.upsample_path(
            e,
            vertical_factor=ctx.cfg.downsample_factor,
            original_length=ctx.cfg.original_height,
        )
        for e in lines
    ]
    (
        ctx.rpe_raw,
        ctx.rpe_guided,
        ctx.rpe_guided_tube_smoothed,
        ctx.rpe_smooth,
    ) = upsampled
    return ctx






# -----------------------------
#  ILM step_ functions
# -----------------------------

def step_ilm_downsample_and_preprocess(ctx: ILMContext) -> ILMContext:
    ctx.img = ctx.original_image.copy()
    # print(f"ILM: shape of img coming in is {ctx.img.shape}")

    d_vertical = ctx.cfg.horizontal_factor   # preserve your original semantics
    d_horizontal = ctx.cfg.vertical_factor

    ctx.d_vertical = float(d_vertical)
    ctx.d_horizontal = float(d_horizontal)

    if d_vertical is not None or d_horizontal is not None:
        ctx.img = cv2.resize(
            ctx.img,
            (0, 0),
            fx=1.0 / d_horizontal,
            fy=1.0 / d_vertical,
        )

    downsampled_img = ctx.img.astype(np.float32)
    ctx.img = gaussian_filter(downsampled_img, sigma=5)

    # print(f"ILM: shape of img after resize is {ctx.img.shape}")
    return ctx


def step_ilm_compute_enhancement(ctx: ILMContext) -> ILMContext:
    d_vertical = ctx.d_vertical if ctx.d_vertical != 0 else 1.0

    vks_false = 2 * round((32 / d_vertical) * ctx.cfg.vks_false_factor)
    vks_true  = 2 * round((128 / d_vertical) * ctx.cfg.vks_true_factor)
    k_rows    = 2 * round((20 / d_vertical) * ctx.cfg.k_rows_factor)
    k_cols    = 2 * round(30 * ctx.cfg.k_cols_factor)

    images = suf.compute_enh_diff(
        ctx.img,
        blur_ksize=ctx.cfg.blur_ksize,
        vks_false=vks_false,
        vks_true=vks_true,
        k_rows=k_rows,
        k_cols=k_cols,
    )
    ctx.enh = images["diff"]
    return ctx


def step_ilm_seed_detection(ctx: ILMContext) -> ILMContext:
    seeds = suf._nms_columnwise(
        ctx.enh,
        radius=ctx.cfg.seeds_radius,
        thresh=ctx.cfg.seeds_thresh,
        value_filter=ctx.cfg.seeds_value_filter,
        narrow_radius_loop=False,
    )
    ctx.seeds = seeds

    edge_raw = suf._hysteresis(
        seeds,
        high=ctx.cfg.hysteresis_high,
        low=ctx.cfg.hysteresis_low,
    )
    ctx.edge_raw = edge_raw
    return ctx


def step_ilm_edge_impute(ctx: ILMContext) -> ILMContext:
    edge = suf.ILM_impute_missing_edges(
        ctx.edge_raw,
        ctx.enh,
        total_seeds=2,
    )
    ctx.edge = edge
    return ctx


def step_ilm_extract_raw(ctx: ILMContext) -> ILMContext:
    ilm_raw = suf._extract_topbottom_line(
        ctx.edge,
        skip_extreme=0,
        direction="top",
    )
    ctx.ilm_raw = ilm_raw
    return ctx


def step_ilm_tube_smoother(ctx: ILMContext) -> ILMContext:
    ilm_smooth, tube_cost_DP_path, tube_cost_raw = suf.tube_smoother_DP(
        img=ctx.enh,
        guide_y=ctx.ilm_raw,
        lambda_step=ctx.cfg.tube_lambda_step,
        max_step=ctx.cfg.tube_max_step,
        sigma_guide=ctx.cfg.tube_sigma_guide,
    )

    ctx.ilm_smooth = ilm_smooth
    ctx.ilm_tube_cost_DP_path = tube_cost_DP_path
    ctx.ilm_tube_cost_raw = tube_cost_raw
    return ctx


def step_ilm_upsample(ctx: ILMContext) -> ILMContext:
    ilm_raw, ilm_smooth = [
        suf.upsample_path(
            e,
            vertical_factor=ctx.cfg.vertical_factor,
            original_length=ctx.cfg.original_height,
        )
        for e in [ctx.ilm_raw, ctx.ilm_smooth]
    ]
    ctx.ilm_raw = ilm_raw
    ctx.ilm_smooth = ilm_smooth
    return ctx


