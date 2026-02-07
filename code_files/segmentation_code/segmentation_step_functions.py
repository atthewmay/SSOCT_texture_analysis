
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # adds Han_AIR/ to path
import code_files.segmentation_code.segmentation_utility_functions as suf
import code_files.segmentation_code.segmentation_plot_utils as spu
import code_files.file_utils as fu
# import code_files.segmentation_utils as su
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Callable, List
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

# type alias for step functions
RPEStepFn = Callable[["RPEContext"], "RPEContext"]
ILMStepFn = Callable[["ILMContext"], "ILMContext"]
C = fu.load_constants()
from joblib import Parallel,delayed




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

# @dataclass(frozen=True)
class HighResConfig:
    which_smoother: str = "hypersmoother"
    down_hblur: Optional[int] = None
    up_hblur: Optional[int] = None
    down_vertical_kernel_size: Optional[int] = None
    up_vertical_kernel_size: Optional[int] = None
    tube_sigma: Optional[int] = 30
    tube_hard_window: Optional[int] = 40
    hysteresis_high: Optional[int] = 0.2
    hysteresis_low: Optional[int] = 0.02

    min_flat_threshold: Optional[int] = 3

    highfreq_down_hblur: Optional[int] = 40
    highfreq_up_hblur: Optional[int] = 50
    highfreq_down_vertical_kernel_size: Optional[int] = 4
    highfreq_up_vertical_kernel_size: Optional[int] = 15

    tube2_sigma: Optional[int] = 15 # Newly set 1/26/25 upon instituting flattenign by RPE. Was 30.
    tube2_hard_window: Optional[int] = 20 
    guide_reward: Optional[int] = 1.2 # Most likey to have to sweep of any parameter


# @dataclass(frozen=True)
class HighResContext: # Just a storage for the higher-resolution rpe processing image steps
    diff_down_up: Optional[np.ndarray] = None
    gaussian_tubed: Optional[np.ndarray] = None
    lower_edge_of_tubed: Optional[np.ndarray] = None
    rpe_refined: Optional[np.ndarray] = None
    rpe_raw: Optional[np.ndarray] = None
    highfreq_diff_down_up: Optional[np.ndarray] = None
    highres_suppressed: Optional[np.ndarray] = None
    guide_image: Optional[np.ndarray] = None
    rpe_refined2: Optional[np.ndarray] = None
    rpe_smooth2: Optional[np.ndarray] = None

@dataclass
class HypersmootherParams: # Just a storage for  The lines by which we flatten during processing pipelines
    hypersmoother_path: Optional[np.ndarray] = None
    hypersmoother_path_extras: Dict[str, np.ndarray] = field(default_factory=dict)
    hypersmoother_shift_y_full: Optional[np.ndarray] = None
    coarse_hypersmoothed_img: Optional[np.ndarray] = None
    hypersmoother_target_y: Optional[np.ndarray] = None
    hypersmoother_y_dp: Optional[np.ndarray] = None # The nonupsampled version

    highres_smoother_path: Optional[np.ndarray] = None
    highres_smoother_shift_y_full: Optional[np.ndarray] = None
    highres_smoother_target_y: Optional[np.ndarray] = None

@dataclass
class RPEContext:
    # inputs
    idx: int
    original_image: np.ndarray
    ONH_region: Any
    cfg: RPEConfig
    ID: Optional[str] = None
    highres_cfg: Optional[HighResConfig] = None
    highres_ctx: Optional[HighResContext] = None
    hypersmoothed_img: np.ndarray = None
    highres_smoothed_img: np.ndarray = None
    hypersmoother_params: HypersmootherParams = field(default_factory=HypersmootherParams)
    img: np.ndarray = None              # working image (may be downsampled)
    downsampled_img: np.ndarray = None              # working image (may be downsampled)
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

    hypersmooth_line: Optional[np.ndarray] = None
    rpe_raw: Optional[np.ndarray] = None
    ilm_margin: Optional[float] = None

    rpe_guided: Optional[np.ndarray] = None
    guided_cost: Optional[np.ndarray] = None
    guided_cost_raw: Optional[np.ndarray] = None

    rpe_guided_tube_smoothed: Optional[np.ndarray] = None
    guided_cost_tube_smoothed: Optional[np.ndarray] = None
    guided_cost_raw_tube_smoothed: Optional[np.ndarray] = None

    rpe_smooth: Optional[np.ndarray] = None
    flat_rpe_smooth: Optional[np.ndarray] = None

    highres_diff_horiz_blur: Optional[np.ndarray] = None
    lower_edge_of_tubed: Optional[np.ndarray] = None
    lower_edge_line: Optional[np.ndarray] = None


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


import pickle
# def ckpt(step, overwrite=False, cache_file = Path(C['root'])/'results/temp_pickle/pipeline_ctx_cache.pickle'):
#     """step is one of the followign step functions
#     use will be to wrap the step function while defining the list of steps like
#     pipeline = [ step1,
#     step2
#     ckpt(step3,overwrite=True),
#     ...
#     ]
#     and then when speeding up, comment out the upper steps. Or could say steps = filter(steps)
#     with def filter s.t. you pop all that are not of type ckpt """
#     def wrapped(ctx):
#         p = Path(cache_file)
#         if p.exists() and not overwrite:
#             return pickle.load(open(p, "rb"))   # <- returns ctx "as of" this point
#         ctx = step(ctx)
#         pickle.dump(ctx, open(p, "wb"))
#         return ctx
#     wrapped._is_ckpt = True
#     wrapped._overwrite = overwrite
#     wrapped._wrapped_step = step
#     return wrapped



class CkptWrap:
    def __init__(self, step, overwrite=False, save_by_ID=False, cache_file=None):
        self.step = step
        self.overwrite = overwrite
        assert cache_file is not None
        self.cache_file = Path(cache_file)

        # keep your tags
        self._is_ckpt = True
        self._overwrite = overwrite
        self._wrapped_step = step
        self.save_by_ID = save_by_ID

    def __call__(self, ctx):
        p = self.cache_file

        if self.save_by_ID:
            # insert _{idx} before suffix
            p = p.with_name(f"{p.stem}_{ctx.ID}{p.suffix}")

        if p.exists() and not self.overwrite:
            with open(p, "rb") as f:
                return pickle.load(f)
        ctx = self.step(ctx)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(ctx, f, protocol=pickle.HIGHEST_PROTOCOL)
        return ctx

def ckpt(step, overwrite=False, save_by_ID=False,cache_file=Path(C['root'])/'results/temp_pickle/pipeline_ctx_cache.pickle'):
    return CkptWrap(step, overwrite=overwrite, save_by_ID=save_by_ID, cache_file=cache_file)


def filter_pipeline(pipeline, start_at=None):
    """
    If start_at is None:
        - find the first ckpt-wrapped step in the list
        - return pipeline starting from that step (so earlier steps won't run)
    If start_at is given (a function object):
        - return pipeline starting at that function
    """
    if start_at is not None:
        i = pipeline.index(start_at)
        return pipeline[i:]

    for i, fn in enumerate(pipeline):
        if getattr(fn, "_is_ckpt", False):
            if getattr(fn,"_overwrite"):
                return pipeline
            return pipeline[i:]
    return pipeline  # no ckpt found -> run all




# -----------------------------
#  RPE step_ functions
# -----------------------------

def step_rpe_init_working(ctx: RPEContext) -> RPEContext:
    """make the .img the working copy"""
    ctx.img  = ctx.original_image.astype(np.float32)  # or .copy() if you prefer
    return ctx

def step_rpe_hypersmoother(ctx: RPEContext) -> RPEContext:
    """run the big coarse smoother as inital preprocess. if using this you have to unsmooth at the end, which is why we store the hypersmooth line. We also adjust the ilm line"""
    ctx.hypersmoother_params.coarse_hypersmoothed_img,ctx.hypersmoother_params.hypersmoother_path,ctx.hypersmoother_params.hypersmoother_y_dp = suf.rpe_hypersmoother_DP(ctx.img,ds_x=8,ds_y=8)
    # """
    # _,ctx.hypersmoother_params.hypersmoother_path_extras['gradient_line'],_ = suf.rpe_hypersmoother_DP(ctx.img,ds_x=2,ds_y=2,preprocess_function=suf.HypersmoothPreprocessors._gradient,preprocess_kwargs={})
    #HYPERSMOOTH DEBUGGING
    """
    AB = spu.ArrayBoard(skip=False,plt_display=False,save_tag="_coarse_hypersmoother_img")
    for d_factor in [2,4,8]:
        for lambda_step in [0,0.05,0.1]:
            for preprocess_fn,kwargs in ((suf.HypersmoothPreprocessors._gblur,{'sigma':4}),
                                        (suf.HypersmoothPreprocessors._gradient,{})):
                coarse_img,_,y_dp = suf.rpe_hypersmoother_DP(ctx.img,ds_x=d_factor,ds_y=d_factor,lambda_step=lambda_step,preprocess_function=preprocess_fn,preprocess_kwargs=kwargs)
        # _ = spu.sweep_to_arrayboard(AB,lambda kw: suf.rpe_hypersmoother_basic_DP,base_kwargs={'img':ctx.img},
        #                         grid = {
        #                                     'rigidity' : [40],
        #                                     'sig' : [4.0],
        #                                     'ds_y' : [4,8],
        #                                     'ds_x' : [4,8],
        #                                     'max_step' : [5],
        #                         })
                AB.add(coarse_img,lines={'DP path':y_dp},title=f"d_factor={d_factor},\nlambda = {lambda_step},\nfn={str(preprocess_fn)}")
    AB.render()

    raise Exception("Intending to end here")
    """
    ctx.hypersmoothed_img,ctx.hypersmoother_params.hypersmoother_shift_y_full,ctx.hypersmoother_params.hypersmoother_target_y = suf.flatten_to_path(ctx.img,ctx.hypersmoother_params.hypersmoother_path)
    # ctx.hypersmoothed_img,ctx.hypersmoother_params = suf.rpe_hypersmoother_basic_DP(ctx.img)
    # ilm_original = ctx.ilm_seg.copy()
    ctx.ilm_seg_flat = suf.warp_line_by_shift(ctx.ilm_seg,ctx.hypersmoother_params.hypersmoother_shift_y_full,direction="to_flat")
    ctx.ilm_seg = ctx.ilm_seg_flat

    ctx.img = ctx.hypersmoothed_img
    # AB = spu.ArrayBoard()
    # AB.add(ctx.original_image,lines={"ilm_original":ilm_original},title = 'original ilm')
    # AB.add(ctx.hypersmoothed_img,lines={"ilm_warped":ctx.ilm_seg},title = 'warped ilm')
    # AB.render()
    return ctx

def step_rpe_highres_smooth(ctx: RPEContext) -> RPEContext:
    """reflatten any needed images by another line. Output the line, and save the params for later unsmoothing
    Specifically this won't modify the old images bc the highres pathway is totally separate from the original pathway."""

    ctx.highres_smoothed_img,ctx.hypersmoother_params.highres_smoother_shift_y_full,ctx.hypersmoother_params.highres_smoother_target_y = suf.flatten_to_path(ctx.original_image,ctx.rpe_smooth)
    ctx.flat_rpe_smooth = suf.warp_line_by_shift(ctx.rpe_smooth,ctx.hypersmoother_params.highres_smoother_shift_y_full,direction="to_flat")
    ctx.img = ctx.highres_smoothed_img

    # AB = spu.ArrayBoard(plt_display=False,save_tag="testing_new_flattener")
    # AB.add(ctx.original_image,lines={'rpe_smooth':ctx.rpe_smooth},title='input')
    # AB.add(ctx.highres_smoothed_img,lines={'rpe_smooth':ctx.flat_rpe_smooth},title='flattened')
    # AB.render()
    return ctx



def step_rpe_unsmooth(ctx: RPEContext) -> RPEContext:
    """undoes the hypersmoothing for the lines and ILM. """
    lines = [
        ('rpe_raw',ctx.rpe_raw),
        ('rpe_guided',ctx.rpe_guided),
        ('rpe_guided_tube_smoothed',ctx.rpe_guided_tube_smoothed),
        ('rpe_smooth',ctx.rpe_smooth),
        ('ilm_seg',ctx.ilm_seg),
        ]
   
    for item in lines:
        name,line = item
        ctx.log_history(f'flat_{name}', line.copy())
    
    unwarped_lines = [suf.warp_line_by_shift(line[1],ctx.hypersmoother_params.hypersmoother_shift_y_full,direction='to_orig') for line in lines] #shoudl refactor
    

    (ctx.rpe_raw,
    ctx.rpe_guided,
    ctx.rpe_guided_tube_smoothed,
    ctx.rpe_smooth,
    ctx.ilm_seg) = unwarped_lines
    return ctx




    # ctx = helperFunctions._unflatten_lines(ctx,)
    # return ctx

def step_rpe_highres_unsmooth(ctx: RPEContext) -> RPEContext:
    """undoes the hypersmoothing for the lines used in the highres pathway. Includes the rpe_smooth which will have been flattened for this path"""
    lines = [
        ('highres_rpe_raw',ctx.highres_ctx.rpe_raw),
        ('highres_rpe_refined',ctx.highres_ctx.rpe_refined),
        ('highres_rpe_refined2',ctx.highres_ctx.rpe_refined2),
        ('highres_rpe_smooth2',ctx.highres_ctx.rpe_smooth2),
        ]

    for item in lines:
        name,line = item
        print(f"logging for flat_{name}")
        ctx.log_history(f'flat_{name}', line.copy())

    unwarped_lines = [suf.warp_line_by_shift(line[1],ctx.hypersmoother_params.highres_smoother_shift_y_full,direction='to_orig') for line in lines] #shoudl refactor

    (ctx.highres_ctx.rpe_raw,
    ctx.highres_ctx.rpe_refined,
    ctx.highres_ctx.rpe_refined2,
    ctx.highres_ctx.rpe_smooth2,
    ) = unwarped_lines

    return ctx


def step_rpe_downsample_and_preprocess(ctx: RPEContext) -> RPEContext:
    """as of 1_3_26, will expect a nondownsampled ILM"""
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
        ctx.ilm_seg = ctx.ilm_seg//ctx.d_vertical

    # print(f"the shape of img after resize is {ctx.img.shape}")
    # ctx.img = ctx.img.astype(np.float32)
    ctx.downsampled_img = gaussian_filter(ctx.img, sigma=5)
    ctx.img = ctx.downsampled_img
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
    I think the fn here is to recover some of the peaks that were oversuppressed. That is my guess. 
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


# A helper function here

def step_rpe_paths_prob_edge(ctx: RPEContext) -> RPEContext:
    """Specifically for the initial coarse RPE segementation we use the ctx.cfg values"""
    prob,edge = helperFunctions._rpe_paths_prob_edge(path_trace_image=ctx.peak_suppressed,
                                         seeds = ctx.seeds,
                                         trace_neighborhood=ctx.cfg.neighbourhood,
                                         hysteresis_high=ctx.cfg.hysteresis_high,
                                         hysteresis_low=ctx.cfg.hysteresis_low,
                                         )
    ctx.prob = prob
    ctx.edge = edge
    return ctx

def step_EZ_rpe_paths_prob_edge(ctx: RPEContext) -> RPEContext:
    """for the refined distinction of EZ vs RPE we will use custom values""" 
    prob,edge = helperFunctions._rpe_paths_prob_edge(path_trace_image=ctx.peak_suppressed,
                                         seeds = ctx.seeds,
                                         trace_neighborhood=ctx.cfg.neighbourhood,
                                         hysteresis_high=ctx.cfg.hysteresis_high,
                                         hysteresis_low=ctx.cfg.hysteresis_low,
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

# def step_rpe_highres_diff_and_tube(ctx: RPEContext) -> RPEContext:
#     which_smoother = ctx.highres_cfg['which_smoother']
#     if which_smoother == "rpe_smooth":
#         flat_img = suf.flatten_to_path(ctx.original_image)
#     elif which_smoother == "hypersmooth_line":
#         flat_img = suf.flatten_to_path(ctx.original_image)

#     diff_down_up,_,_,_ = suf.diff_boundary_enhance_and_blur_horiz(ctx.hypersmoothed_img,**diff_boundary_enhance_kwargs)

#     gaussian_tubed = apply_gaussian_tube_mul(diff_down_up,rpe_smooth_guide,highres_tube_sigma,gain=1,hard_window=40)
#     lower_edge_of_tubed = _normalized_axial_gradient(gaussian_tubed,vertical_kernel_size=4,dark2bright=True)

def step_rpe_highres_diff_enh(ctx: RPEContext) -> RPEContext:
    """Using the flattened image, we take down adn up grads w/ highres. Bc we now work in flattened domain,
    we use a straight horizontal blur rather than downward horiz blur. Ultimately use the diff betwen the down-up grad horiz blurred images. 
    Returns the ctx, but also processes the diffed """
    if getattr(ctx.highres_cfg,'down_vertical_kernel_size') is None:
        #"""assume they are all none and set these attrs"""
        ctx.highres_cfg.down_hblur=40
        ctx.highres_cfg.up_hblur=50
        ctx.highres_cfg.down_vertical_kernel_size=25
        ctx.highres_cfg.up_vertical_kernel_size=15
   
    # The following aplies a coarser kernel than teh highest freq
    ctx.highres_ctx.diff_down_up,diff_up_down,hblur_down,hblur_up = suf.diff_boundary_enhance_and_blur_horiz(ctx.img, # Use the latest image
                                                                        down_hblur=ctx.highres_cfg.down_hblur,
                                                                        up_hblur=ctx.highres_cfg.up_hblur,
                                                                        down_vertical_kernel_size=ctx.highres_cfg.down_vertical_kernel_size,
                                                                        up_vertical_kernel_size=ctx.highres_cfg.up_vertical_kernel_size)

    ctx.highres_ctx.gaussian_tubed = suf.apply_gaussian_tube_mul(ctx.highres_ctx.diff_down_up,ctx.flat_rpe_smooth, # changing to the flat version
                                             sigma = ctx.highres_cfg.tube_sigma,
                                             gain=1,
                                             hard_window = ctx.highres_cfg.tube_hard_window)
    ctx.highres_ctx.lower_edge_of_tubed = suf._normalized_axial_gradient(ctx.highres_ctx.gaussian_tubed,vertical_kernel_size=4,dark2bright=True) # This should be moved to config if it's swept ever, but really just sharpens the bottom edge. 
    return ctx

def step_rpe_highres_peak_suppress_to_rpe_refined(ctx: RPEContext) -> RPEContext:
    """get peaks, lay seeds, trace seeds, select bottom line-> ctx.highres_ctx.RPE_raw, 
    peak_suppress upward to get rid of EZ, DP on this grad image -> ctx.highres_ctx.rpe_refined"""

    peaks = suf.peakSuppressor.EZ_RPE_peak_suppression_pipeline(ctx.highres_ctx.lower_edge_of_tubed,ctx.highres_ctx.lower_edge_of_tubed,ilm_line=None,**{'sigma':1,
                                                                                                                        'peak_distance':5,
                                                                                                                        'peak_prominance':0.05}) # again if sweep later, refactor into config

    seeds = suf.peaks_to_seeds(peaks,ctx.highres_ctx.lower_edge_of_tubed.shape[0]) # convert o the boolean matrix

    # OUT_peak_img = spu.overlay_peaks_on_image(lower_edge_of_tubed,peaks=peaks) # debug only, keepign in sweeper function

    probs,edge = helperFunctions._rpe_paths_prob_edge(path_trace_image=ctx.highres_ctx.lower_edge_of_tubed,
                                                        seeds = seeds,
                                                        trace_neighborhood=1,
                                                        hysteresis_high=ctx.highres_cfg.hysteresis_high,
                                                        hysteresis_low=ctx.highres_cfg.hysteresis_low)

    rpe_raw = suf._extract_topbottom_line(edge,skip_extreme=0,direction='bottom')
    ctx.highres_ctx.highres_suppressed = suf.peakSuppressor.suppress_above_below_line(ctx.highres_ctx.lower_edge_of_tubed,rpe_raw,factor=0.5) # refactor suppression factor if sweeping
    cost = suf.modify_cost_with_ONH_info(1-ctx.highres_ctx.highres_suppressed,ONH_region=ctx.ONH_region,ONH_value_factor=0.5) # shape mismatch likely 

    # AB = spu.ArrayBoard(plt_display=False,save_tag="cost_checker")
    # AB.add(1-suppressed)
    # AB.add(cost)
    # AB.render()

    rpe_refined,_ = suf.run_DP_on_cost_matrix(cost,max_step=2,lambda_step=0.5)
    ctx.highres_ctx.rpe_raw = rpe_raw
    ctx.highres_ctx.rpe_refined = rpe_refined
    return ctx


def step_rpe_highres_higher_res_gradient_guided_DP_to_rpe_refined2(ctx: RPEContext) -> RPEContext:
    """executes the final steps in the rpe search: at this point we have the RPE mostly identified, with a few jumps to choroid or EZ.
    1. truncate to keep flat portions of RPE only as guide-points (bc we want to eliminate jumps). 
    2. highest-resolution gradient image obtained with small kernels
    3. tube this
    4. apply the truncated guide
    5. DP w/ a (sadly sensitive) guide reward. At worst this should snap back to RPE refined, which is better than our ctx.rpe_smooth """
    truncated_rpe_refined = suf.keep_only_flat_segments(ctx.highres_ctx.rpe_refined,slope_tol=0.5,min_len=ctx.highres_cfg.min_flat_threshold) # Really this isn't quite what we want. 

    highfreq_diff_down_up,_,_,_ = suf.diff_boundary_enhance_and_blur_horiz(ctx.img, # use the current image here
                                                                        down_hblur=ctx.highres_cfg.highfreq_down_hblur,
                                                                        up_hblur=ctx.highres_cfg.highfreq_up_hblur,
                                                                        down_vertical_kernel_size=ctx.highres_cfg.highfreq_down_vertical_kernel_size,
                                                                        up_vertical_kernel_size=ctx.highres_cfg.highfreq_up_vertical_kernel_size)


    tubed_highres_grad = suf.apply_gaussian_tube_mul(highfreq_diff_down_up,ctx.highres_ctx.rpe_refined,
                                                     sigma=ctx.highres_cfg.tube2_sigma,
                                                     gain=1,
                                                     hard_window=ctx.highres_cfg.tube2_hard_window) # Tighten the sigma
    lower_edge_of_highres_grad_tubed = suf._normalized_axial_gradient(tubed_highres_grad,vertical_kernel_size=4,dark2bright=True)
    guide_image = suf.apply_guideline_discount(lower_edge_of_highres_grad_tubed,guideline=truncated_rpe_refined,factor=ctx.highres_cfg.guide_reward,band=1)
    cost = suf.modify_cost_with_ONH_info(1-guide_image,ctx.ONH_region,0.5)
    rpe_refined2,_ = suf.run_DP_on_cost_matrix(cost,max_step=2,lambda_step=0.3) # Already a pretty dang good output

    """Debugging the final step here: Looks like a tighter sigma for the win, and likely the entire second refinement step in this step function is unneeded. 
    AB = spu.ArrayBoard(plt_display=False,save_tag=f'sweep final guide {ctx.ID}',ncols_max=5)
    for sigma in [10,20,30]:
        for guide_reward in [1,1.5]:
            tubed_highres_grad = suf.apply_gaussian_tube_mul(highfreq_diff_down_up,ctx.highres_ctx.rpe_refined,
                                                            sigma=sigma,
                                                            gain=1,
                                                            hard_window=ctx.highres_cfg.tube2_hard_window) # Tighten the sigma
            lower_edge_of_highres_grad_tubed = suf._normalized_axial_gradient(tubed_highres_grad,vertical_kernel_size=4,dark2bright=True)
            guide_image = suf.apply_guideline_discount(lower_edge_of_highres_grad_tubed,guideline=truncated_rpe_refined,factor=guide_reward,band=1)
            cost = suf.modify_cost_with_ONH_info(1-guide_image,ctx.ONH_region,0.5)
            rpe_refined2,_ = suf.run_DP_on_cost_matrix(cost,max_step=2,lambda_step=0.3) # Already a pretty dang good output
            AB.add(tubed_highres_grad,title='tubed highres grad')
            AB.add(lower_edge_of_highres_grad_tubed,title='lower_edge_of_highres_grad_tubed')
            AB.add(cost,title='raw cost')
            AB.add(cost,lines={'rpe_refined1':ctx.highres_ctx.rpe_refined,'rpe_refined2':rpe_refined2},title=f'sigma={sigma},guide_rew={guide_reward}')
            AB.add(ctx.img,lines={'rpe_smooth':ctx.rpe_smooth,'rpe_refined1':ctx.highres_ctx.rpe_refined,'rpe_refined2':rpe_refined2},title='input img')
    AB.render()

    raise Exception("intending to stop")
    """


    
    # ctx.highres_ctx.guide_image  = guide_image 
    ctx.highres_ctx.guide_image  = cost 
    ctx.highres_ctx.highfreq_diff_down_up = highfreq_diff_down_up
    ctx.highres_ctx.rpe_refined2 = rpe_refined2
    ctx.highres_ctx.rpe_smooth2 = suf.smooth_rpe_line(rpe_refined2,rigidity=25)

    return ctx






def step_rpe_highres_flat_guided(ctx: RPEContext) -> RPEContext:
    """1_3_26 attempt to pull the real RPE off the flattened image now that we have our guiding signal. 
    We'll do things like the down and up gradiants, zoomed in on the small tube of allowed region, via the usual tube-smoothing method. """

    # should break this into its component steps i think. 
    # print("yes gonna try to run the AB via debugging")
    # AB = spu.ArrayBoard(skip=False,plt_display=False,dpi=600)
    # # Idea, could even try toggling which input image we give, flattening according to RPE seg could be a good idea
    # spu.sweep_to_arrayboard(AB,suf.debug_high_res_tube_rpe_finder,base_kwargs={'flat_img':ctx.hypersmoothed_img,
    #                                                                            'rpe_smooth_guide':ctx.rpe_smooth,
    #                                                                            'highres_tube_sigma':30},
    #                         grid = {
    #                             'down_hblur':[40],
    #                             'up_hblur':[50],
    #                             'down_vertical_kernel_size':[15,25],
    #                             'up_vertical_kernel_size':[15],
    #                         }
    # )

    #                                     # down_vertical_kernel_size = 25,
    #                                     # down_blur_ksize = 20,
    #                                     # # up_params = {'vertical_kernel_size':15,'blur_ksize':40},
    #                                     # up_vertical_kernel_size = 15,
    #                                     # up_blur_ksize = 40,
    #                                     # down_hblur = 40,
    #                                     # up_hblur = 50,


    # AB.render()


    # diff_be_kwargs = {'down_hblur':[40],
    #             'up_hblur':[50],
    #             'down_vertical_kernel_size':[15,25],
    #             'up_vertical_kernel_size':[15],}

 
    from datetime import datetime
    # AB_suppressed_up=spu.ArrayBoard(dpi=500,plt_display=False,skip=False,save_tag=f'_suppressed_up_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}') #don't overwrite for a bit
    AB_suppressed_up=spu.ArrayBoard(dpi=800,plt_display=False,skip=False,save_tag=f'big_sweep_work_idx={ctx.ID}') #don't overwrite for a bit
    _ = spu.sweep_to_arrayboard(AB_suppressed_up,suf.sweeper_fn,
                                base_kwargs={'images_dict':{'flat_img':ctx.hypersmoothed_img},
                                             'rpe_smooth_guide':ctx.rpe_smooth,
                                             'really_big_sweep':True,
                                        'down_hblur':40,
                                        'up_hblur':50,
                                        # 'down_vertical_kernel_size':2,4,25,
                                        'down_vertical_kernel_size':25,
                                        'up_vertical_kernel_size':15,
                                        'peak_prominance':0.05,
                                        # "lambda_step":0.1,

                                        "hysteresis_high":0.2,
                                        # "hysteresis_low":0.05,
                                        "hysteresis_low":0.02,



                                        'highfreq_down_hblur':40,
                                        'highfreq_up_hblur':50,
                                        'highfreq_down_vertical_kernel_size':4,
                                        'highfreq_up_vertical_kernel_size':15,

                                             },
                                grid = {
                                        # 'down_hblur':[40],
                                        # 'up_hblur':[50],
                                        # # 'down_vertical_kernel_size':[2,4,25],
                                        # 'down_vertical_kernel_size':[25],
                                        # 'up_vertical_kernel_size':[15],
                                        # 'factor':[0.3],
                                        # 'peak_prominance':[0.05],
                                        # "hysteresis_high":[0.2],
                                        # "hysteresis_low":[0.01],
                                        # "lambda_step":[0.1],
                                        "lambda_step":[0.5],
                                        'factor':[0.5],

 # Really big sweep params

                                        'tube2_sigma' : [30],
                                        'tube2_hard_window':[10,20],


                                        'min_flat_threshold':[3],
                                        'guide_reward' : [1.2,1.4],
                                        'lambda_step2' : [0.3],



                                        },
                                        nworkers=2
                                        )
    AB_suppressed_up.render()
    # raise Exception
   


    # Focusing on the big sweep as of 1/10/26
    # ctx.highres_diff_horiz_blur,ctx.lower_edge_of_tubed,ctx.lower_edge_line = suf.debug_high_res_tube_rpe_finder(ctx.hypersmoothed_img,
    #                                                                                                              original_image=ctx.original_image,
    #                                                                                                              rpe_hypersmooth_params = ctx.hypersmoother_params,
    #                                                                                                              rpe_smooth_guide=ctx.rpe_smooth,
    #                                                                                                              highres_tube_sigma=30)

    return ctx




class helperFunctions(object):
    """These are static methods that don't take ctx themselves, but can be better parameterized by ctx.
      They are still compliations of the suf functions, so I felt they belonged here"""
    @staticmethod
    def _rpe_paths_prob_edge(path_trace_image,seeds,trace_neighborhood,hysteresis_high,hysteresis_low,nworkers=1) -> RPEContext:
        paths = suf._trace_paths(
            path_trace_image,
            seeds,
            trace_neighborhood,
        )
        prob = suf._probability_image(paths, path_trace_image.shape)

        # multiply prob by intensity (your MULT_PROB_TIMES_INTENSITY block)
        prob *= path_trace_image
        prob /= (prob.max() + 1e-6)

        edge = suf._hysteresis(
            prob,
            high=hysteresis_high,
            low=hysteresis_low,
        )
        return prob,edge

    @staticmethod
    def _unflatten_lines(ctx: RPEContext,unflattening_shift):
        lines = [
            ('rpe_raw',ctx.rpe_raw),
            ('rpe_guided',ctx.rpe_guided),
            ('rpe_guided_tube_smoothed',ctx.rpe_guided_tube_smoothed),
            ('rpe_smooth',ctx.rpe_smooth),
            ('ilm_seg',ctx.ilm_seg),
            ]

        if getattr(ctx.highres_ctx,'rpe_refined2') is not None: # recall it's gonna default to none
            lines += [
                ('highres_rpe_raw',ctx.highres_ctx.rpe_raw),
                ('highres_rpe_refined',ctx.highres_ctx.rpe_refined),
                ('highres_rpe_refined2',ctx.highres_ctx.rpe_refined2),
                ]
        
        for item in lines:
            name,line = item
            ctx.log_history(f'flat_{name}', line.copy())
        
        unwarped_lines = [suf.warp_line_by_shift(line[1],unflattening_shift,direction='to_orig') for line in lines] #shoudl refactor
        
        if getattr(ctx.highres_ctx,'rpe_refined2') is not None:
            (ctx.rpe_raw,
            ctx.rpe_guided,
            ctx.rpe_guided_tube_smoothed,
            ctx.rpe_smooth,
            ctx.ilm_seg,
            ctx.highres_ctx.rpe_raw,
            ctx.highres_ctx.rpe_refined,
            ctx.highres_ctx.rpe_refined2,
            ) = unwarped_lines
        else:
            (ctx.rpe_raw,
            ctx.rpe_guided,
            ctx.rpe_guided_tube_smoothed,
            ctx.rpe_smooth,
            ctx.ilm_seg) = unwarped_lines
        return ctx



# -------------------
# GS Functions
# ---------
def step_rpe_highres_GS(ctx: RPEContext) -> RPEContext:
    """perform a single GS"""
    AB = spu.ArrayBoard(plt_display=False,save_tag="GS_testing")
    value = np.unique(ctx.flat_rpe_smooth)
    assert len(value)==1
    value = value[0]
    out = suf.gs_single_surface_pymaxflow(ctx.highres_ctx.highres_suppressed,row_range=[value-200,value+200],smoothness=0) # 
    AB.add(ctx.highres_ctx.diff_down_up,title = 'lower edge of tubed')
    AB.add(ctx.highres_ctx.lower_edge_of_tubed,lines = {'rpe_flat':ctx.flat_rpe_smooth},title = 'lower edge of tubed')
    AB.add(ctx.highres_ctx.highres_suppressed,title = 'input')
    AB.add(ctx.highres_ctx.highres_suppressed,lines = {'gs_seg_line':out},title='output')
    AB.render()
    raise Exception

def step_rpe_highres_DP2(ctx: RPEContext) -> RPEContext:
    """perform a single GS"""
    AB = spu.ArrayBoard(plt_display=False,save_tag="GS_testing")
    value = np.unique(ctx.flat_rpe_smooth)
    assert len(value)==1
    value = value[0]
    radius = 40
    print(value-radius)
    img_band = ctx.highres_ctx.highres_suppressed[int(value-radius):int(value+radius),:]
    # _ = spu.sweep_to_arrayboard(AB,fn = suf.run_two_surface_DP,
    #                         base_kwargs={'cost1':1-img_band,'cost2':1-img_band, 'lambda1':0.01, 'lambda2':0.01, 'max_step1':1, 'max_step2':1},
    #                         grid={'dmin':[4,10,20]},
    #                         )
    lines_to_add = {}
    def loop_contents(d_min):
        y1, y2 = suf.run_two_surface_DP(1-img_band, 1-img_band, dmin=d_min, dmax=50, lambda1=0.01, lambda2=0.01,max_step1=1,max_step2=1)
        return d_min,y1,y2
    
    results = Parallel(n_jobs=-1)(delayed(loop_contents)(d) for d in [4,10])
    for r in results:
        lines_to_add[f'y1_dmin={r[0]}']=r[1]
        lines_to_add[f'y2_dmin={r[0]}']=r[2]
    

    AB.add(img_band,lines=lines_to_add,title = 'img_band_with_lines')
    AB.add(ctx.highres_ctx.diff_down_up,title = 'lower edge of tubed')
    AB.add(ctx.highres_ctx.lower_edge_of_tubed,lines = {'rpe_flat':ctx.flat_rpe_smooth},title = 'lower edge of tubed')
    for k,v in lines_to_add.items():
        lines_to_add[k] = v-radius+value
    AB.add(ctx.highres_ctx.highres_suppressed,lines = lines_to_add,title = 'input')
    AB.render()
    raise Exception


















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


