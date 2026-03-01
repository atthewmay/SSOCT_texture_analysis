
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

from textwrap import fill



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
    lower_edge_of_highres_grad_tubed: Optional[np.ndarray] = None # highfreq version
    guide_image: Optional[np.ndarray] = None
    rpe_refined2: Optional[np.ndarray] = None
    rpe_smooth2: Optional[np.ndarray] = None

@dataclass
class twoLayerDPContext: # Just a storage for the higher-resolution rpe processing image steps
    debug_bool: Optional[bool] = True
    y1: Optional[np.ndarray] = None
    y2: Optional[np.ndarray] = None
    img_band: Optional[np.ndarray] = None
    y1_rescaled: Optional[np.ndarray] = None
    y2_rescaled: Optional[np.ndarray] = None
    debug: dict = field(default_factory=dict)


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
    two_layer_dp_ctx: Optional[twoLayerDPContext] = None
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

    rpe_enh_DP_cost_raw: Optional[np.ndarray] = None

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
    ID: Optional[str] = None
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


    hypersmoothed_img: Optional[np.ndarray] = None
    hypersmoother_params: HypersmootherParams = field(default_factory=HypersmootherParams)

    peak_suppressed: Optional[np.ndarray] = None

    inv_cost: Optional[np.ndarray] = None
    thinline_inv_cost: Optional[np.ndarray] = None

    penultimate_DP_cost: Optional[np.ndarray] = None
    penultimate_DP_darkness_barrier_img: Optional[np.ndarray] = None

    # final_DP_thinline_cost: Optional[np.ndarray] = None
    # final_DP_thinline_darkness_barrier_img: Optional[np.ndarray] = None

    ilm_smooth_thinline: Optional[np.ndarray] = None

    # Final refining steps
    DP_refining_tube: Optional[np.ndarray] = None
    DP_final_cost: Optional[np.ndarray] = None
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



# def step_rpe_unsmooth(ctx: RPEContext) -> RPEContext:
#     """undoes the hypersmoothing for the lines and ILM. """
#     lines = [
#         ('rpe_raw',ctx.rpe_raw),
#         ('rpe_guided',ctx.rpe_guided),
#         ('rpe_guided_tube_smoothed',ctx.rpe_guided_tube_smoothed),
#         ('rpe_smooth',ctx.rpe_smooth),
#         ('ilm_seg',ctx.ilm_seg),
#         ]
   
#     for item in lines:
#         name,line = item
#         ctx.log_history(f'flat_{name}', line.copy())
    
#     unwarped_lines = [suf.warp_line_by_shift(line[1],ctx.hypersmoother_params.hypersmoother_shift_y_full,direction='to_orig') for line in lines] #shoudl refactor
    

#     (ctx.rpe_raw,
#     ctx.rpe_guided,
#     ctx.rpe_guided_tube_smoothed,
#     ctx.rpe_smooth,
#     ctx.ilm_seg) = unwarped_lines
#     return ctx

def step_rpe_unsmooth(ctx: RPEContext) -> RPEContext:
    """Undo hypersmoothing warp for whatever lines exist (and ILM if present)."""
    names = ["rpe_raw", "rpe_guided", "rpe_guided_tube_smoothed", "rpe_smooth", "ilm_seg"]

    shift = ctx.hypersmoother_params.hypersmoother_shift_y_full

    for name in names:
        if not hasattr(ctx, name):
            continue
        line = getattr(ctx, name)
        if line is None:
            continue

        # optional logging if your ctx has it
        if hasattr(ctx, "log_history") and callable(getattr(ctx, "log_history")):
            ctx.log_history(f"flat_{name}", line.copy())

        setattr(
            ctx,
            name,
            suf.warp_line_by_shift(line, shift, direction="to_orig"),
        )

    return ctx



    # ctx = helperFunctions._unflatten_lines(ctx,)
    # return ctx

def step_rpe_highres_unsmooth(ctx: RPEContext) -> RPEContext:
    """undoes the hypersmoothing for the lines used in the highres pathway. Includes the rpe_smooth which will have been flattened for this path"""
    base_lines = [
        ('highres_rpe_raw',ctx.highres_ctx.rpe_raw),
        ('highres_rpe_refined',ctx.highres_ctx.rpe_refined),
        ('highres_rpe_refined2',ctx.highres_ctx.rpe_refined2),
        ('highres_rpe_smooth2',ctx.highres_ctx.rpe_smooth2),
        ]

    # two_dp_names = ['y1', 'y2', 'y1_rescaled', 'y2_rescaled']
    two_dp_names = ['y1_rescaled', 'y2_rescaled']
    # print(ctx.two_layer_dp_ctx)
    for n in two_dp_names:
        if hasattr(ctx.two_layer_dp_ctx,n):
            # print(f"should be adding {n}")
            base_lines.append((n,getattr(ctx.two_layer_dp_ctx,n)))
        
    # print([e[0] for e in base_lines])

    for item in base_lines:
        name,line = item
        # print(f"logging for flat_{name}")
        ctx.log_history(f'flat_{name}', line.copy())

    # unwarped_lines = [suf.warp_line_by_shift(line[1],ctx.hypersmoother_params.highres_smoother_shift_y_full,direction='to_orig') for line in lines] #shoudl refactor
    shift = ctx.hypersmoother_params.highres_smoother_shift_y_full
    unwarped = {
            name: suf.warp_line_by_shift(line, shift, direction="to_orig")
            for name, line in base_lines
    }

    # (ctx.highres_ctx.rpe_raw,
    # ctx.highres_ctx.rpe_refined,
    # ctx.highres_ctx.rpe_refined2,
    # ctx.highres_ctx.rpe_smooth2,
    # ) = unwarped_lines

    # assign back only the ones that belong to highres_ctx
    ctx.highres_ctx.rpe_raw      = unwarped["highres_rpe_raw"]
    ctx.highres_ctx.rpe_refined  = unwarped["highres_rpe_refined"]
    ctx.highres_ctx.rpe_refined2 = unwarped["highres_rpe_refined2"]
    ctx.highres_ctx.rpe_smooth2  = unwarped["highres_rpe_smooth2"]

        # if include_two_layer and getattr(ctx, "two_layer_dp_ctx", None) is not None:
    for n in two_dp_names:
        setattr(ctx.two_layer_dp_ctx, n, unwarped[n])

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

def step_rpe_compute_enhancement2(ctx: RPEContext) -> RPEContext:
    """2/28/26, now that we flatten w/ a different lowres rpe approx, we will trial w/o the aggressive enh_diff 
    (formerly removed choroid signal, which now isn't as distracting, bc we don't try to one-shot the RPE). Will go ahead and combine peak suppression here"""
    d_vertical = ctx.d_vertical if ctx.d_vertical != 0 else 1.0
    vks_false = 2 * round((32 / d_vertical) * ctx.cfg.vks_false_factor)
    enh_f = suf._boundary_enhance(
        ctx.img,
        vertical_kernel_size=vks_false,
        dark2bright=False,
        blur_ksize=ctx.cfg.blur_ksize,
    )
    ctx.enh = enh_f

    ctx.peak_suppressed = suf.peakSuppressor.peak_suppression_pipeline(
        ctx.enh,
        ctx.enh,
        ctx.ilm_seg,
        suppression_factor=0.5,
        third_peak_margin=30,
    )

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
    pre_suppressed_recalculated = suf.recalculate_single_seeded_cols(ctx.seeds,ctx.peak_suppressed,ctx.enh_f) #ah, this is bringing in old info that was too-aggressively removed usgng the downward horizontal blur
    # Taking that info from the enh_f image
    ctx.log_history('pre_suppressed_recalculated',pre_suppressed_recalculated)
    ctx.peak_suppressed = suf.peakSuppressor.peak_suppression_pipeline( #not sure why we are running this a second time, as i expect the same result... No, they are differrent, but I'm not sure why
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

def step_rpe_DP_on_enh_1(ctx: RPEContext) -> RPEContext:
    """testing a different endpoint for the lower-res pathway where we instead just run a DP after suppressing below the ctx.ilm_seg"""
    inv_cost = suf.normalize_image(ctx.peak_suppressed)
    # cost = 1-inv_cost
    AB = spu.ArrayBoard(skip=True,plt_display=False,save_tag=f"testing new DP on enh, ID={ctx.ID}")
    # AB.add(inv_cost,lines={'ilm_seg':ctx.ilm_seg},title='inv_cost in')
    # for sigma in [1,5,10]:
    #     for gain in [1,3]:
    #         processed_inv_cost = suf.apply_gaussian_tube_suppression(inv_cost,ctx.ilm_seg,sigma=sigma,gain=gain)
    #         AB.add(processed_inv_cost,title = f'inv_cost with sigma={sigma},gain={gain}')
    # AB.render()

    processed_inv_cost = suf.apply_gaussian_tube_suppression(inv_cost,ctx.ilm_seg,sigma=10,gain=1)
    # AB.add(processed_inv_cost,title = 'with ILM suppressed')
    cost = 1-processed_inv_cost
    # print(f"ctx onh region is shape {ctx.ONH_region.shape}, like {ctx.ONH_region}")
    cost = suf.modify_cost_with_ONH_info(cost,ctx.ONH_region,ONH_value_factor=0.5)
    ctx.rpe_enh_DP_cost_raw = cost # assign this for plotting
    # AB.add(cost,title='cost in')
    # for l in [0,0.3,0.5,1]:
        # DP_out,_ = suf.run_DP_on_cost_matrix(cost,max_step=1,lambda_step=l,ONH_region=ctx.ONH_region,lambda_step_in_ONH_region=0.0001)
        # AB.add(inv_cost,lines = {"DP line":DP_out},title=f"lambda step = {l}")
    # AB.render()
    ctx.rpe_smooth,_ = suf.run_DP_on_cost_matrix(cost,max_step=1,lambda_step=0.01,ONH_region=ctx.ONH_region,lambda_step_in_ONH_region=0.0001)

    return ctx

def step_rpe_DP_on_enh_2_debug(ctx: RPEContext) -> RPEContext:
    """testing a different endpoint for the lower-res pathway where we instead just run a DP after suppressing below the ctx.ilm_seg"""
    inv_cost = suf.normalize_image(ctx.peak_suppressed)
    # cost = 1-inv_cost
    AB = spu.ArrayBoard(skip=False,plt_display=False,save_tag=f"testing new DP on enh, ID={ctx.ID}")
    AB.add(inv_cost,lines={'ilm_seg':ctx.ilm_seg},title='inv_cost in')
    processed_inv_cost = suf.apply_gaussian_tube_suppression(inv_cost,ctx.ilm_seg,sigma=10,gain=1)
    AB.add(processed_inv_cost,title = 'with ILM suppressed')
    cost = 1-processed_inv_cost
    # print(f"ctx onh region is shape {ctx.ONH_region.shape}, like {ctx.ONH_region}")
    cost = suf.modify_cost_with_ONH_info(cost,ctx.ONH_region,ONH_value_factor=0.5)
    AB.add(cost,title='cost in')

    ctx.rpe_enh_DP_cost_raw = cost # assign this for plotting

    def loop_contents(combo):
        # Bcs,barrier = suf.calculate_darkness_barrier_and_Bcs(cost,combo.get('t',None))
        norm_cost = suf.normalize_image_per_column(cost)
        Bcs,barrier = suf.calculate_darkness_barrier_and_Bcs(norm_cost,t=combo.get('t',None),p=combo['p'])
        DP_path,_ = suf.run_DP_on_cost_matrix(cost,max_step=1,lambda_step=0.01,ONH_region=ctx.ONH_region,lambda_step_in_ONH_region=0.001,dbf=combo['dbf'],Bcs=Bcs)
        return DP_path,barrier,combo
    combos=[]
    for dbf in [5]:
        # for alpha in [2,8]:
        for t in [0.5]:
            for p in [1,2]:
                    # combos.append({'l':l,'dbf':dbf,'alpha':alpha})
                combos.append({'dbf':dbf,'t':t,'p':p})
    results = Parallel(n_jobs=8)(delayed(loop_contents)(c) for c in combos)
    for r in results:
        AB.add(r[1],title=f'barrier raw for {r[2]}')
        AB.add(inv_cost,lines={"DP_path":r[0]},title=f'{r[2]}')
    AB.render()
    raise Exception("Done experimenetal plotting")


def step_rpe_DP_on_enh_2(ctx: RPEContext) -> RPEContext:
    """testing a different endpoint for the lower-res pathway where we instead just run a DP after suppressing below the ctx.ilm_seg"""
    inv_cost = suf.normalize_image(ctx.peak_suppressed)
    processed_inv_cost = suf.apply_gaussian_tube_suppression(inv_cost,ctx.ilm_seg,sigma=10,gain=1)
    cost = 1-processed_inv_cost
    cost = suf.modify_cost_with_ONH_info(cost,ctx.ONH_region,ONH_value_factor=0.5)
    ctx.rpe_enh_DP_cost_raw = cost # assign this for plotting
    norm_cost = suf.normalize_image_per_column(cost)
    Bcs,barrier = suf.calculate_darkness_barrier_and_Bcs(norm_cost,t=0.5,p=1)
    ctx.rpe_smooth,_ = suf.run_DP_on_cost_matrix(cost,max_step=1,lambda_step=0.01,ONH_region=ctx.ONH_region,lambda_step_in_ONH_region=0.001,dbf=5,Bcs=Bcs)

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
    ctx.rpe_smooth = suf.smooth_rpe_line(
        ctx.rpe_guided_tube_smoothed,
        rigidity=ctx.cfg.rigidity,
    )



    # spu.save_image_exploration(ctx.img, rpe_guided_tube_smoothed, pickle_save=False) # won't actually run
    return ctx


# def step_rpe_upsample(ctx: RPEContext) -> RPEContext:
#     lines = [
#         ctx.rpe_raw,
#         ctx.rpe_guided,
#         ctx.rpe_guided_tube_smoothed,
#         ctx.rpe_smooth,
#     ]
#     upsampled = [
#         suf.upsample_path(
#             e,
#             vertical_factor=ctx.cfg.downsample_factor,
#             original_length=ctx.cfg.original_height,
#         )
#         for e in lines
#     ]
#     (
#         ctx.rpe_raw,
#         ctx.rpe_guided,
#         ctx.rpe_guided_tube_smoothed,
#         ctx.rpe_smooth,
#     ) = upsampled
#     return ctx

def step_rpe_upsample(ctx: RPEContext) -> RPEContext:
    """Upsample whatever RPE-ish paths exist on ctx back to original height."""
    names = ["rpe_raw", "rpe_guided", "rpe_guided_tube_smoothed", "rpe_smooth"]

    for name in names:
        if hasattr(ctx, name):
            e = getattr(ctx, name)
            if e is None:
                continue
            setattr( ctx, name,
                suf.upsample_path( e, vertical_factor=ctx.cfg.downsample_factor, original_length=ctx.cfg.original_height,),
            )
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
    ctx.highres_ctx.lower_edge_of_highres_grad_tubed = lower_edge_of_highres_grad_tubed 
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
# Double DP 2 layer function
#---------------------
def step_rpe_highres_DP_two_layer(ctx: RPEContext) -> RPEContext:
    """
    This is the real function!
    Debug step: run 2-surface DP sweeps on a band around the flattened line,
    visualize in ArrayBoard.
    """
    import code_files.segmentation_code.two_surface_utils as tsu

    AB = spu.ArrayBoard(skip=True,plt_display=False, ncols_max=5, save_tag=f"DP 2-layer step for {ctx.ID}")

    # --- flatten center (your code assumes flat_rpe_smooth is constant)
    offset_value = np.unique(ctx.flat_rpe_smooth)
    try:
        assert len(offset_value) == 1
    except:
        print(f"For some reason, the len(offset_value) != 1. Instead, its {len(offset_value)}")
    offset_value = float(offset_value[0])

    img_band_radius = 30
    r0 = int(max(0, offset_value - img_band_radius))
    r1 = int(min(ctx.highres_ctx.lower_edge_of_tubed.shape[0], offset_value + img_band_radius))

    img_full = ctx.highres_ctx.lower_edge_of_tubed
    img_band = img_full[r0:r1, :]

    # “gradient image as source” (simple vertical abs-grad; replace with your preferred)
    # dy = np.diff(img_band, axis=0, prepend=img_band[[0], :])
    # grad_band = np.abs(dy)

    # Costs: use same pipeline but different sources
    cost_int = tsu.make_cost_from_img(img_band, mode="inv_colmax")

    # --- build a param sweep

    # determined that 
    # 2/12 params
    # base = dict(
    #     dmin=10, dmax=40,
    #     max_step1=1, max_step2=1,
    #     lambda1=0.6, lambda2=0.6,

    #     # new behavior (NMS-based)
    #     prefer_lower_on_single=True,
    #     single_kappa=0.1,          # still the main knob
    #     peak_sigma=1,
    #     peak_distance= 5,
    #     peak_prominance = 0.05, # again if sweep later, refactor into config

    #     # optional extras
    #     sep_gamma=0.0,
    #     depth_a1=0.0,
    #     depth_a2=0.0,
    # )

    # 2/16 params
    base = dict(
        dmin=10, dmax=40,
        max_step1=1, max_step2=1,
        lambda1=0.6, lambda2=0.6,

        # new behavior (NMS-based)
        prefer_lower_on_single=True,
        single_kappa=0.4,          # still the main knob
        kappa_mode = 'reweight',
        darkness_barrier_factor = 0,
        peak_sigma=1,
        peak_distance= 5,
        peak_prominance = 0.05, # again if sweep later, refactor into config

        # optional extras
        sep_gamma=0.0,
        depth_a1=0.0,
        depth_a2=0.0,
    )
 


    # --- run sweeps on intensity and gradient sources
    y1, y2, dbg = tsu.run_two_surface_DP(cost_int, cost_int, return_debug=True, **base)
    def rescale(y):  
        return y - img_band_radius + offset_value
    y1_rescaled,y2_rescaled = rescale(y1),rescale(y2)
    # --- visuals (your usual context panels)
    AB.add(ctx.original_image, title="original")
    AB.add(ctx.highres_ctx.diff_down_up, title="input (diff_down_up)")
    AB.add(ctx.highres_ctx.lower_edge_of_tubed, lines={"rpe_flat": ctx.flat_rpe_smooth}, title="lower_edge_of_tubed + flat")

    # band source images
    AB.add(img_band, lines={'y1':y1,'y2':y2},title="img_band (intensity)")
    # AB.add(ctx.highres_ctx.diff_down_up, lines={"current final": ctx.highres_ctx.rpe_refined2}, title="current performance")
    AB.add(ctx.highres_smoothed_img, lines={"current final": ctx.highres_ctx.rpe_refined2,'y1':y1_rescaled,'y2':y2_rescaled}, title="current performance")
    AB.render()
    ctx.two_layer_dp_ctx.img_band = img_band
    ctx.two_layer_dp_ctx.y1 = y1
    ctx.two_layer_dp_ctx.y2 = y2
    ctx.two_layer_dp_ctx.y1_rescaled = y1_rescaled
    ctx.two_layer_dp_ctx.y2_rescaled = y2_rescaled
    if ctx.two_layer_dp_ctx.debug_bool:
        ctx.two_layer_dp_ctx.debug = dbg
    return ctx

def step_rpe_endpoint_plot(ctx: RPEContext) -> RPEContext:
    """do some plotting to summarize the pathway"""
    AB = spu.ArrayBoard(skip=False,plt_display=False,save_tag=f"final_plot_step_{ctx.ID}")
    # Add the RPE plots
    # if 'gradient_line' in ctx_rpe.hypersmoother_params.hypersmoother_path_extras:
        # AB.add(ctx_rpe.original_image,lines={'hypersmoothed':ctx_rpe.hypersmoother_params.hypersmoother_path,'line via upgrad':ctx_rpe.hypersmoother_params.hypersmoother_path_extras['gradient_line']},title="original")
    AB.add(ctx.original_image,lines={'hypersmoothed':ctx.hypersmoother_params.hypersmoother_path},title="original")
    AB.add(ctx.hypersmoother_params.coarse_hypersmoothed_img,title="hypersmooth_coarse")
    AB.add(ctx.hypersmoothed_img,title="hypersmoothed_img")
    AB.add(ctx.downsampled_img,title="downsampled_img")
    if ctx.enh_f is not None:
        AB.add(ctx.enh_f,title="enh_f")
    if ctx.enh is not None:
        AB.add(ctx.enh,title="enh_diff (pre suppressed)")
    if hasattr(ctx,'original_peak_suppressed') and ctx.original_peak_suppressed is not None:
        print("also plotting the original peak suppressed")
        AB.add(ctx.original_peak_suppressed,title="original peak_suppresed")
    AB.add(ctx.peak_suppressed,title="peak_suppressed")
    if hasattr(ctx,'rpe_enh_DP_cost_raw') and ctx.rpe_enh_DP_cost_raw is not None: 
        AB.add(ctx.rpe_enh_DP_cost_raw,title="rpe_enh_DP_cost_raw")
    if hasattr(ctx,'prob') and ctx.prob is not None:
        AB.add(ctx.prob,title="prob")
    if hasattr(ctx,'edge') and ctx.edge is not None:
        AB.add(ctx.edge,title="edge")
    if hasattr(ctx,'guided_cost_raw') and ctx.guided_cost_raw is not None:
        AB.add(ctx.guided_cost_raw,title="guided_cost_raw")
    if hasattr(ctx,'guided_cost_raw_tubed_smoothed') and ctx.guided_cost_raw_tubed_smoothed is not None:
        AB.add(ctx.guided_cost_raw_tube_smoothed,title="guided_cost_raw_tube_smoothed")
    AB.add(ctx.original_image,lines = {"hypersmoothed":ctx.hypersmoother_params.hypersmoother_path, "rpe_smooth":ctx.rpe_smooth},title="original with rpe_smooth")
    # now the highres stuff
    if not hasattr(ctx.highres_ctx,'diff_down_up') or ctx.highres_ctx.diff_down_up is None:
        raise Exception("We are intentionally terminating rpe plotting early as it appears the highres pipeline was not actually run")
    AB.add(ctx.highres_ctx.diff_down_up,title="buggy highfreq_diff_down_up")
    AB.add(ctx.highres_ctx.lower_edge_of_tubed,title="lower_edge_of_tubed")
    if hasattr(ctx,'highres_suppressed'): # A logged history version
        AB.add(ctx.highres_suppressed,title="tubed_and_suppressed")
    AB.add(ctx.highres_ctx.lower_edge_of_tubed,lines = {'rpe_flat':ctx.flat_rpe_smooth,"rpe_raw":ctx.flat_highres_rpe_raw,"rpe_refined":ctx.flat_highres_rpe_refined},title="lower_edge_of_tubed with rpe raw (lower edge) and refined with DP")
    
    AB.add(ctx.highres_ctx.lower_edge_of_tubed,lines = {"rpe_raw":ctx.flat_highres_rpe_raw,"rpe_refined":ctx.flat_highres_rpe_refined},title="lower_edge_of_tubed with rpe raw (lower edge) and refined with DP")
    AB.add(ctx.highres_ctx.highfreq_diff_down_up,title="highest-res gradient")
    AB.add(ctx.highres_ctx.guide_image,title="DP_guide_img")

    # Now add the two-layer stuff
    if ctx.two_layer_dp_ctx.debug:
        print("adding debug info")
        dbg = ctx.two_layer_dp_ctx.debug
        peaks = dbg['peaks']
        peak_img = spu.overlay_peaks_on_image(ctx.two_layer_dp_ctx.img_band,peaks)
        AB.add(peak_img,  title='2-layer DP peak img')
        # AB.add(ctx.two_layer_dp_ctx.debug,lines = {'y1':ctx.two_layer_dp_ctx.y1,'y2': ctx.two_layer_dp_ctx.y2},title='img_band')
    AB.add(ctx.two_layer_dp_ctx.img_band,lines = {'y1':ctx.two_layer_dp_ctx.y1,'y2': ctx.two_layer_dp_ctx.y2},title='img_band')

    # Summarizing original imgs
    AB.add(ctx.original_image,lines = {"hypersmoothed":ctx.hypersmoother_params.hypersmoother_path,
                                            "rpe_smooth":ctx.rpe_smooth,
                                            "rpe_refined1":ctx.highres_ctx.rpe_refined,
                                            "rpe_refined2":ctx.highres_ctx.rpe_refined2,
                                            "rpe_smooth2":ctx.highres_ctx.rpe_smooth2,
                                            },title="Original with all lines")

    AB.add(ctx.original_image,lines = { 

                                            "rpe_smooth":ctx.rpe_smooth,
                                        "two_layer_y1":ctx.two_layer_dp_ctx.y1_rescaled,
                                        "two_layer_y2":ctx.two_layer_dp_ctx.y2_rescaled,
                                        "rpe_smooth2":ctx.highres_ctx.rpe_smooth2,
                                        },title="Original with two-layer lines")
    AB.render()
    # raise Exception("endpoint plotting complete. Terminating here!")
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
    AB = spu.ArrayBoard(plt_display=False,ncols_max=5, save_tag=f"DP2 testing for {ctx.ID}")
    value = np.unique(ctx.flat_rpe_smooth)
    assert len(value)==1
    value = value[0]
    radius = 30
    print(value-radius)
    # img_to_process = ctx.highres_ctx.highres_suppressed
    img_to_process = ctx.highres_ctx.lower_edge_of_tubed
    img_band = img_to_process[int(value-radius):int(value+radius),:]

    lines_to_add = {}
    def loop_contents(d_min):
        y1, y2 = suf.run_two_surface_DP(1-img_band, 1-img_band, dmin=10, dmax=40, lambda1=0.01, lambda2=0.01,max_step1=1,max_step2=1)
        return d_min,y1,y2
    
    results = Parallel(n_jobs=8)(delayed(loop_contents)(d) for d in [10])
    y1,y2 = results[0][1],results[0][2]
    for r in results:
        lines_to_add[f'y1_dmin={r[0]}']=r[1].copy()
        lines_to_add[f'y2_dmin={r[0]}']=r[2].copy()
    

    AB.add(ctx.original_image,title = 'original')
    AB.add(ctx.highres_ctx.highfreq_diff_down_up,title = 'highest_freq_diff')
    AB.add(ctx.highres_ctx.diff_down_up,title = 'input')
    AB.add(ctx.highres_ctx.lower_edge_of_tubed,lines = {'rpe_flat':ctx.flat_rpe_smooth},title = 'lower edge of tubed')
    AB.add(ctx.highres_ctx.highres_suppressed,title = 'highres suppressed')
    from copy import deepcopy
    lines_to_add_adjusted = deepcopy(lines_to_add)
    for k,v in lines_to_add_adjusted.items():
        lines_to_add_adjusted[k] = v-radius+value
    AB.add(img_band,lines=lines_to_add,title = 'img_band_with_lines')
    AB.add(img_to_process,lines = lines_to_add_adjusted,title = 'line generating image with lines')
    AB.add(ctx.highres_ctx.diff_down_up,lines = lines_to_add_adjusted,title = 'input with lines')
    AB.add(ctx.highres_ctx.diff_down_up,lines = {'current final':ctx.highres_ctx.rpe_refined2},title = 'current performance')
    

    # Eval goodness of fit
    # Get my peaks
    peaks = suf.peakSuppressor.EZ_RPE_peak_suppression_pipeline(ctx.highres_ctx.lower_edge_of_tubed,ctx.highres_ctx.lower_edge_of_tubed,ilm_line=None,**{'sigma':1,
                                                                                                                        'peak_distance':5,
                                                                                                                        'peak_prominance':0.05}) # again if sweep later, refactor into config

    OUT_peak_img = spu.overlay_peaks_on_image(ctx.highres_ctx.lower_edge_of_tubed,peaks=peaks) # debug only, keepign in sweeper function
    AB.add(OUT_peak_img,title="peaks here")

    from scipy.ndimage import median_filter,uniform_filter1d
    # for title,x in [["suppressed",ctx.highres_ctx.highres_suppressed],["tubed_only",ctx.highres_ctx.lower_edge_of_tubed]]:
    for title,x in [["tubed_only",ctx.highres_ctx.lower_edge_of_tubed]]:
        den = np.max(x, axis=0, keepdims=True)   # shape (1, 512)
        normed_eval_img = x / (den + 1e-8)
        gof,parts = suf.two_line_gof_from_peaks_and_gap(normed_eval_img,y1-radius+value,y2-radius+value,peaks=peaks,dmin=10,return_parts=True,peak_tol=3)
        

        def plot_parts_norm(ax,parts,filter_type,norm=True):
            for k,v in parts.items():
                if len(v)!=512:
                    continue
                if norm:
                    v_norm = v/np.max(v)
                else:
                    v_norm = v
                if filter_type == 'mean':
                    y_med = uniform_filter1d(v_norm, size=50, mode="nearest")
                elif filter_type == 'median':
                    y_med = median_filter(v_norm, size=25, mode="nearest")
                    print("using median filter")
                else:
                    raise Exception("must supply filter type")
                ax.plot(y_med,alpha=0.5,linewidth=0.8,label=f"{k}, range=[{round(np.min(v),2)},{round(np.max(v),2)}]")
            ax.legend(fontsize=4, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))

        def plot_simple(ax,plot_dict):
            for k,v in plot_dict.items():
                ax.plot(v,alpha=0.5,linewidth=0.8,label=f"{k}")
                ax.legend(fontsize=4, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
        # for i,subparts_k in enumerate([[ 'peak_score', 'pin_score', 'dist1', 'dist2'],

        # def compute_product(parts,k1,k2,filter_type,filter_width,return_metric=False):
        #     """modifies parts dict in place"""
        #     if filter_type == 'median':
        #         filt = lambda v_norm,w=filter_width: median_filter(v_norm, size=w, mode="nearest")
        #     elif filter_type == 'mean':
        #         filt = lambda v_norm,w=filter_width: uniform_filter1d(v_norm, size=w, mode="nearest")
        #     f1 = filt(parts[k1])
        #     f2 = filt(parts[k2])
        #     prod = f1*f2
        #     if return_metric:
        #         return prod
        #     parts[f"{k1}_*_{k2}"] = prod

        # compute_product(parts,'peak_score','intensity_score','median',50)

        for i,subparts_k in enumerate([[ 'peak_score', 'dist1', 'dist2'],
                    # [ 'parallel_score', 'gap', 'gap_std'],
                    [ 'intensity1', 'intensity2', 'intensity_score']]):
            subparts = {k: parts[k] for k in subparts_k}
            # filter_type = 'median' if i == 0 else 'mean'
            filter_type = 'median'
            AB.add_plot(lambda ax, sp=deepcopy(subparts),ft=filter_type: plot_parts_norm(ax, sp,ft),
                        title=f'gof parts with {title}, filter={filter_type}')
        
        peak_inten,peak_inten_narrowed,mask = suf.GOFProcessing.EZ_inclusion_selection(parts)
        plot_dict = {'peak_inten':peak_inten, 'peak_inten_narrowed':peak_inten_narrowed, 'mask': mask}
        AB.add_plot(lambda ax: plot_simple(ax,plot_dict),
                    title=f'gof processed')
            


            

    AB.render()
    raise Exception


###############
def step_rpe_highres_DP2_debug(ctx: RPEContext) -> RPEContext:
    """
    Debug step: run 2-surface DP sweeps on a band around the flattened line,
    visualize in ArrayBoard.
    """
    import code_files.segmentation_code.segmentation_plot_utils as spu
    import numpy as np
    import code_files.segmentation_code.two_surface_utils as tsu

    AB = spu.ArrayBoard(plt_display=False, ncols_max=5, save_tag=f"DP2 dbf debug testing for {ctx.ID}")

    # --- flatten center (your code assumes flat_rpe_smooth is constant)
    offset_value = np.unique(ctx.flat_rpe_smooth)
    assert len(offset_value) == 1
    offset_value = float(offset_value[0])

    img_band_radius = 30
    r0 = int(max(0, offset_value - img_band_radius))
    r1 = int(min(ctx.highres_ctx.lower_edge_of_tubed.shape[0], offset_value + img_band_radius))

   # --- build a param sweep
    AB.add(ctx.original_image, title="original")
    AB.add(ctx.highres_ctx.diff_down_up, title="input (diff_down_up)")
    # AB.add(ctx.highres_ctx.lower_edge_of_tubed, lines={"rpe_flat": ctx.flat_rpe_smooth}, title="lower_edge_of_tubed + flat")
    AB.add(ctx.highres_ctx.lower_edge_of_tubed, title="lower_edge_of_tubed + flat")
    AB.add(ctx.highres_ctx.lower_edge_of_highres_grad_tubed, title="highfreq lower edge of tube flat")
    

    base = dict(
        dmin=10, dmax=40,
        max_step1=1, max_step2=1,
        lambda1=0.01, lambda2=0.01,

        # new behavior (NMS-based)
        prefer_lower_on_single=True,
        single_kappa=0.2,          # still the main knob
        peak_sigma=1,
        peak_distance= 5,
        peak_prominance = 0.05, # again if sweep later, refactor into config

        # optional extras
        sep_gamma=0.0,
        depth_a1=0.0,
        depth_a2=0.0,
    )

    
    """
    param_list = []
    for dmin in [10]:
        # for lam in [0.2,0.4,0.6,1]:
        # for img_title,img_full in [('lower res tube',ctx.highres_ctx.lower_edge_of_tubed),('higher res tube',ctx.highres_ctx.lower_edge_of_highres_grad_tubed)]: # A failed experiement for now!
        for img_title,img_full in [('lower res tube',ctx.highres_ctx.lower_edge_of_tubed)]: 
            for peak_prominance in [0.05]:
                for lam in [0.6]:
                    for cost_mode in ['inv_colmax']: # seems better?
                    # for cost_mode in ['inv_colmax','inv_global']:
                        for kappa_mode in ['reweight']:
                        # for kappa_mode in ['abs_dist','reweight']:
                            # for kappa in [0.4,0.5,0.6]:
                            for kappa in [0.4,0.6]:
                                for barrier_cost_params in [{'alpha':a} for a in [2,4,6]]:
                                    # for dbf in [0,0.1,0.3,0.5]:
                                    for dbf in [0,0.3,0.4,0.7]:
                            # for nms_thresh in [0.01,0.1]:
                            #     for nms_radius in [4,10]:

                                        # img_full = ctx.highres_ctx.lower_edge_of_tubed
                                        # img_full = img_tuple[1]
                                        img_band = img_full[r0:r1, :]
                                        # Costs: use same pipeline but different sources
                                        cost_int = tsu.make_cost_from_img(img_band, mode=cost_mode)


                                        p = dict(base)
                                        p['cost1'] = cost_int
                                        p['cost2'] = cost_int
                                        p["dmin"] = dmin
                                        p["lambda1"] = lam
                                        p["lambda2"] = lam
                                        p["darkness_barrier_factor"] = dbf
                                        p["kappa_mode"] = kappa_mode
                                        p["cost_mode"] = cost_mode
                                        p["input_img"] = img_title
                                        p["peak_prominance"] = peak_prominance
                                        p["barrier_cost_params"] = barrier_cost_params
                                        # p["max_step1"] = max_step
                                        # p["max_step2"] = max_step
                                        p["single_kappa"] = kappa
                                        param_list.append(p)
    """


    # Trying to debug the dbf and exponential form
    param_list = []
    for dmin in [10]:
        # for lam in [0.2,0.4,0.6,1]:
        # for img_title,img_full in [('lower res tube',ctx.highres_ctx.lower_edge_of_tubed),('higher res tube',ctx.highres_ctx.lower_edge_of_highres_grad_tubed)]: # A failed experiement for now!
        for img_title,img_full in [('lower res tube',ctx.highres_ctx.lower_edge_of_tubed)]: 
            for peak_prominance in [0.05]:
                for lam in [0.6]:
                    for cost_mode in ['inv_colmax']: # seems better?
                    # for cost_mode in ['inv_colmax','inv_global']:
                        for kappa_mode in ['reweight']:
                        # for kappa_mode in ['abs_dist','reweight']:
                            # for kappa in [0.4,0.5,0.6]:
                            for kappa in [0.6]:
                                for barrier_cost_params in [{'alpha':a} for a in [2,6,10,20]]:
                                    # for dbf in [0,0.1,0.3,0.5]:
                                    for dbf in [0.5]:
                            # for nms_thresh in [0.01,0.1]:
                            #     for nms_radius in [4,10]:

                                        # img_full = ctx.highres_ctx.lower_edge_of_tubed
                                        # img_full = img_tuple[1]
                                        img_band = img_full[r0:r1, :]
                                        # Costs: use same pipeline but different sources
                                        cost_int = tsu.make_cost_from_img(img_band, mode=cost_mode)


                                        p = dict(base)
                                        p['cost1'] = cost_int
                                        p['cost2'] = cost_int
                                        p["dmin"] = dmin
                                        p["lambda1"] = lam
                                        p["lambda2"] = lam
                                        p["darkness_barrier_factor"] = dbf
                                        p["kappa_mode"] = kappa_mode
                                        p["cost_mode"] = cost_mode
                                        p["input_img"] = img_title
                                        p["peak_prominance"] = peak_prominance
                                        p["barrier_cost_params"] = barrier_cost_params
                                        # p["max_step1"] = max_step
                                        # p["max_step2"] = max_step
                                        p["single_kappa"] = kappa
                                        param_list.append(p)

    # likley good params for two algo
    """
    base = dict(
        dmin=10, dmax=40,
        max_step1=1, max_step2=1,
        lambda1=0.6, lambda2=0.6,

        # new behavior (NMS-based)
        prefer_lower_on_single=True,
        single_kappa=0.4,          # still the main knob
        kappa_mode = 'reweight',
        darkness_barrier_factor = 0,
        peak_sigma=1,
        peak_distance= 5,
        peak_prominance = 0.05, # again if sweep later, refactor into config

        # optional extras
        sep_gamma=0.0,
        depth_a1=0.0,
        depth_a2=0.0,
    )
    """
 



        
    # param_list = []
    # for dmin in [10]:
    #     # for lam in [0.2,0.4,0.6,1]:
    #     for img_title,img_full in [('lower res tube',ctx.highres_ctx.lower_edge_of_tubed),('higher res tube',ctx.highres_ctx.lower_edge_of_highres_grad_tubed)]:
    #         for lam in [0.1]:
    #             for cost_mode in ['inv_colmax']: # seems better?
    #             # for cost_mode in ['inv_colmax','inv_global']:
    #                 for kappa_mode in ['reweight']:
    #                 # for kappa_mode in ['abs_dist','reweight']:
    #                     for kappa in [0.05]:
    #                 # for dbf in [0.1,0.2,0.3,0.4,0.5]:
    #                 # for nms_thresh in [0.01,0.1]:
    #                 #     for nms_radius in [4,10]:

    #                         # img_full = ctx.highres_ctx.lower_edge_of_tubed
    #                         # img_full = img_tuple[1]
    #                         img_band = img_full[r0:r1, :]
    #                         # Costs: use same pipeline but different sources
    #                         cost_int = tsu.make_cost_from_img(img_band, mode=cost_mode)


    #                         p = dict(base)
    #                         p['cost1'] = cost_int
    #                         p['cost2'] = cost_int
    #                         p["dmin"] = dmin
    #                         p["lambda1"] = lam
    #                         p["lambda2"] = lam
    #                         # p["darkness_barrier_factor"] = dbf
    #                         p["kappa_mode"] = kappa_mode
    #                         p["cost_mode"] = cost_mode
    #                         p["input_img"] = img_title

    #                         # p["max_step1"] = max_step
    #                         # p["max_step2"] = max_step
    #                         p["single_kappa"] = kappa
    #                         param_list.append(p)





    # --- run sweeps on intensity and gradient sources
    # results_int = tsu.sweep_two_surface_dp(cost_int, cost_int, param_list, n_jobs=1)
    results_int = tsu.sweep_two_surface_dp(param_list, n_jobs=8)
    # results_grad = sweep_two_surface_dp(cost_grad, cost_grad, param_list, n_jobs=8)
    AB.add(img_band, title="img_band (intensity)")
    # AB.add(1-cost_int, title="1-cost calc from img band")

    # --- visuals (your usual context panels)

    # AB.add(ctx.highres_ctx.lower_edge_of_tubed, lines={"rpe_flat": ctx.flat_rpe_smooth}, title="lower_edge_of_tubed + flat")
    # AB.add(ctx.highres_ctx.highres_suppressed, title="highres_suppressed")

    # band source images

    # overlay best few results (by final cost)
    # results_int_sorted = sorted(results_int, key=lambda r: r["debug"].get("final_cost", np.inf))
    results_int_sorted = results_int
    # results_grad_sorted = sorted(results_grad, key=lambda r: r["debug"].get("final_cost", np.inf))

    tsu.add_dp_sweep_to_arrayboard(
        AB,
        img_band=1-cost_int, # A better version
        # img_band=img_band,
        img_full=img_full,
        band_top=r0,
        results=results_int_sorted,
        title_prefix="INT",
        max_overlays=6,
        offset_value=offset_value,
        img_band_radius=img_band_radius,
        relevant_names=["lambda2",'kappa_mode',"single_kappa",'cost_mode','input_img','darkness_barrier_factor','barrier_cost_params','peak_prominance']
    )

    # add_dp_sweep_to_arrayboard(
    #     AB,
    #     img_band=grad_band,
    #     img_full=img_full,
    #     band_top=r0,
    #     results=results_grad_sorted,
    #     title_prefix="GRAD",
    #     max_overlays=6,
    # )

    # Compare to current performance
    AB.add(ctx.highres_ctx.diff_down_up, lines={"current final": ctx.highres_ctx.rpe_refined2}, title="current performance")

    AB.render()
    raise Exception("DP2 debug stop")
###############


class doublePeakProcessors():
    """likely too much a pain, too brittle"""
    @staticmethod
    def keep_confluent_regions(metric,threshold,percentile,required_width):
        """keep those pixels where the percentile quantile of requrieted_width around them is above threhsold."""
        # fill here
        pass

    @staticmethod
    def zero_narrow(metric,zeroing_metric,low_threshold,low_percentile):
        """we have now selected regions for inclusion based on some metric, but i think the intensity1 metric
        will drop close to zero for spurious selections (choroid grabs). I would intend to decrease regions next to very low regions. 
        Like take a window of some size: if the say 6th percentile (one above low percentile) is within that window is below some 'low_threshold', then that entire window gets cut by some factor or something.
        This would effectively shink peaks of spurious inclusion"""
        # fill here
        pass

    @staticmethod
    def EZ_inclusion_selection(parts,threshold,percentile,required_width):
        """We will include as EZ/RPE regions (double DP), places where the say 5th percnetile (quantile) of the composite score (metric)
        is > some threshold for a span ≥ required_width.  The idea here is to cut down adjacent to regions of lower signal strength of zero"""
        EZ_selection_metric = compute_product(parts,'peak_score','intensity_score','median',50,True)
        EZ_selection_metric = zero_narrow(EZ_selection_metric,parts['intensity1'],low_threshold=0.1,low_percentile=0.05)

        true_false_regions = keep_confluent_regions(EZ_selection_metric,threshold,percentile,required_width)
        return  true_false_regions 



def step_rpe_highres_grad_testing(ctx: RPEContext) -> RPEContext:
    """sadly this is a failed experiment. MAkes sense. The upward grad picks up the lower edge of bright lines. this gets both the choroid and the RPE.
    no free lunch here. """
    """testing some better processing"""
    from textwrap import fill

    img = ctx.highres_smoothed_img # the orignal but flattened

    def zero_above_line(img2d: np.ndarray, y_line: np.ndarray):
        """
        img2d: (H, W)
        y_line: (W,) integer y for each column x
        sets img2d[y < y_line[x], x] = 0
        """
        H, W = img2d.shape
        y_line = np.asarray(y_line)
        y_line = np.clip(y_line.astype(int), 0, H)   # allow H => zero entire column above H (i.e. nothing)

        rows = np.arange(H)[:, None]                 # (H,1)
        mask_above = rows < y_line[None, :]          # (H,W)
        img2d[mask_above] = 0
        return img2d

    def to01(img: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        img = np.asarray(img)
        mn = np.nanmin(img)
        mx = np.nanmax(img)
        return (img - mn) / (mx - mn + eps)

    def loop_contents(d):
        enh_up = suf._boundary_enhance(img,d['kernel_size'],dark2bright=True,blur_ksize=d['blur_ksize'])
        # blurred_img = suf._blur_image(enh_up,d['blur_k_x'],d['blur_k_y'])
        blurred_img = suf.downward_horizontal_blur2(enh_up,d['blur_k_x'],d['blur_k_y'])
        blurred_img = to01(blurred_img)
        blurred_img_zeroed = zero_above_line(blurred_img,ctx.highres_ctx.rpe_smooth2)
        source_img = to01(ctx.highres_ctx.diff_down_up )
        diff_img = source_img - blurred_img_zeroed
        diff_img[diff_img<0] = 0
        diff_img /= (diff_img.max() + 1e-6)

        return d,enh_up,blurred_img,diff_img

        
    
    results = Parallel(n_jobs=2)(delayed(loop_contents)({'kernel_size':ks,'blur_ksize':bs,'blur_k_x':bkx,'blur_k_y':bky}) 
                                 for ks in [50] for bs in [50] for bkx in [30,50,100] for bky in [20,30,100])
                                #  for ks in [40] for bs in [50] for bkx in [20] for bky in [10])
    AB = spu.ArrayBoard(plt_display=False,save_tag=f'other grad testing {ctx.ID}')
    AB.add(ctx.highres_smoothed_img,title='original')
    AB.add(ctx.highres_smoothed_img,lines={'rpe_smooth2':ctx.highres_ctx.rpe_smooth2,
                                           'rpe_smooth':ctx.flat_rpe_smooth},title='original with line')

    AB.add(ctx.highres_ctx.diff_down_up,title='input to second round')
    AB.add(ctx.highres_ctx.diff_down_up,lines={'rpe_smooth2':ctx.highres_ctx.rpe_smooth2,
                                           'rpe_smooth':ctx.flat_rpe_smooth},title='input to second round')
    # AB.add(ctx.highres_ctx.highfreq_diff_down_up,lines={'rpe_smooth2':ctx.highres_ctx.rpe_smooth2,
                                        #    'rpe_smooth':ctx.flat_rpe_smooth},title='input to second round')
    # for r in results:
    #     d = r[0]
    #     AB.add(r[1],lines={'rpe_smooth2':ctx.highres_ctx.rpe_refined2},title=f'down_kb={r[0]}')
    #     AB.add(r[2],lines={'rpe_smooth2':ctx.highres_ctx.rpe_refined2},title=f'up_kb={r[0]}')
    #     AB.add(r[3],lines={'rpe_smooth2':ctx.highres_ctx.rpe_refined2},title=f'left_kb={r[0]}')
    #     AB.add(r[4],lines={'rpe_smooth2':ctx.highres_ctx.rpe_refined2},title=f'right_kb={r[0]}')

    for r in results:
        d = r[0]
        AB.add(r[1],lines={'rpe_smooth2':ctx.highres_ctx.rpe_refined2},title=fill(f'enh_up. d={d}',width=25))
        AB.add(r[2],lines={'rpe_smooth2':ctx.highres_ctx.rpe_refined2},title=fill(f'blurred. d={d}',width=25))
        AB.add(r[3],lines={'rpe_smooth2':ctx.highres_ctx.rpe_refined2},title=fill(f'Diff. d={d}',width=25))
    AB.render()

    return ctx















# -----------------------------
#  ILM step_ functions
# -----------------------------

def step_ilm_hypersmoother(ctx: ILMContext) -> ILMContext:
    """run the big coarse smoother as inital preprocess. if using this you have to unsmooth at the end, which is why we store the hypersmooth line. We also adjust the ilm line"""
    ctx.hypersmoother_params.coarse_hypersmoothed_img,ctx.hypersmoother_params.hypersmoother_path,ctx.hypersmoother_params.hypersmoother_y_dp = suf.rpe_hypersmoother_DP(ctx.original_image,ds_x=8,ds_y=8)
    ctx.hypersmoothed_img,ctx.hypersmoother_params.hypersmoother_shift_y_full,ctx.hypersmoother_params.hypersmoother_target_y = suf.flatten_to_path(ctx.original_image,ctx.hypersmoother_params.hypersmoother_path)
    return ctx


def step_ilm_downsample_and_preprocess(ctx: ILMContext) -> ILMContext:
    if ctx.hypersmoothed_img is not None:
        ctx.img = ctx.hypersmoothed_img.copy()
    else:
        print("using the old step_ilm_downsample_and_preprocess where we directly downsampling the original image")
        ctx.img = ctx.original_image.copy()
    # print(f"ILM: shape of img coming in is {ctx.img.shape}")

    d_vertical = ctx.cfg.horizontal_factor   # preserve your original semantics
    d_horizontal = ctx.cfg.vertical_factor

    ctx.d_vertical = float(d_vertical)
    ctx.d_horizontal = float(d_horizontal)

    if d_vertical is not None or d_horizontal is not None:
        ctx.img = cv2.resize( ctx.img, (0, 0), fx=1.0 / d_horizontal, fy=1.0 / d_vertical,)
        if ctx.ONH_region is not None:
            if hasattr(ctx.ONH_region, "compute"):
                ctx.ONH_region = ctx.ONH_region.compute()
            ctx.ONH_region = cv2.resize( ctx.ONH_region, (0, 0), fx=1.0 / d_horizontal, fy=1.0 / d_vertical,)


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

def step_ilm_peak_suppression(ctx: ILMContext) -> ILMContext:
    """applies peak suppression below first peak. Will be questionable in cases of """
    # AB = spu.ArrayBoard(plt_display=False,save_tag=f'ilm peak suppression test')
    # for sf in [0,0.3,0.5]:
    ctx.peak_suppressed = suf.peakSuppressor.peak_suppression_pipeline(ctx.enh,ctx.enh,ilm_line=None,use_third_peak=False,
                                                suppression_factor=0.1,)
        # AB.add(peak_suppressed,title=f'suppression_factor = {sf}')
    # AB.render()
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

def step_ilm_DP_debug(ctx: ILMContext) -> ILMContext:
    """2/27/26, to follow after some basic peak suppression, to run the ILM"""
    inv_cost = suf.normalize_image(ctx.peak_suppressed,zero_min=True)
    cost = 1-inv_cost
    cost = suf.modify_cost_with_ONH_info(cost,ctx.ONH_region,ONH_value_factor=0.5)

    AB = spu.ArrayBoard(plt_display=False,save_tag=f"DP ILM testing for ID = {ctx.ID}")
    AB.add(ctx.original_image,title=f'original_image')
    AB.add(ctx.enh,title=f'pre-peak-suppressed')
    AB.add(inv_cost,title=f'inv_cost (post peak sup)')
    AB.add(cost,title=f'cost')

    # AB.add(inv_cost_thinline,title=f'inv_cost_thinline (post peak sup)')
    # AB.add(cost_thinline,title=f'cost_thinline')


    def loop_contents(combo):
        # Bcs,barrier = suf.calculate_darkness_barrier_and_Bcs(cost,combo.get('t',None))
        # if combo.get('norm_mode') == 'columnwise':

        if combo['cost_name'] == 'cost':
            cost_in = cost
        elif combo['cost_name'] == "cost_thinline":
            inv_cost_thinline = suf._normalized_axial_gradient(inv_cost.copy(),vertical_kernel_size=combo['vks'],dark2bright=False) # This should be moved to config if it's swept ever, but really just sharpens the bottom edge. 
            cost_thinline = 1-inv_cost_thinline
            cost_in = suf.modify_cost_with_ONH_info(cost_thinline,ctx.ONH_region,ONH_value_factor=0.5)
        else:
            raise Exception("filed to supply proper cost name")
        norm_cost = suf.normalize_image_per_column(cost_in.copy())
        # else:
            # norm_cost = cost.copy()
        Bcs,barrier = suf.calculate_darkness_barrier_and_Bcs(norm_cost,t=combo.get('t',None),p=combo['p'])
        # DP_path,_ = suf.run_DP_on_cost_matrix(cost_in,max_step=3,lambda_step=combo['l'],ONH_region=ctx.ONH_region,lambda_step_in_ONH_region=0.001,dbf=combo['dbf'],Bcs=Bcs)
        DP_path,_ = suf.run_DP_on_cost_matrix(cost_in,max_step=3,lambda_step=0.01,ONH_region=ctx.ONH_region,lambda_step_in_ONH_region=0.001,dbf=combo['dbf'],Bcs=Bcs)
        return DP_path,barrier,combo,cost_in

    combos=[]
    # for norm_mode in ['columnwise']:
    # for name,cost in ['columnwise']:
    for cost_name in ["cost","cost_thinline"]:
        for vks in [4,12,25]:
            # for l in [0.01]:
            for dbf in [5]:
                # for alpha in [2,8]:
                for t in [0.6]:
                    # for p in [0.1,1]:
                    for p in [1]:
                # combos.append({'l':l,'dbf':dbf,'alpha':alpha})
                    # combos.append({"norm_mode":norm_mode,'l':l,'dbf':dbf,'t':t,'p':p})
                        combos.append({"cost_name":cost_name,'dbf':dbf,'t':t,'p':p,'vks':vks})

    results = Parallel(n_jobs=8)(delayed(loop_contents)(c) for c in combos)
    for r in results:
        AB.add(r[1],title=f'barrier raw for {r[2]}')
        AB.add(1-r[-1],lines={"DP_path":r[0]},title=fill(f'1-cost for {r[2]}',width=25))
    AB.render()
    raise Exception


# def step_ilm_DP(ctx: ILMContext) -> ILMContext:
#     """2/27/26, to follow after some basic peak suppression, to run the ILM"""
#     t = 0.6
#     # t = 0.6
#     p=1
#     dbf=5
#     lambda_step=0.01

#     inv_cost1 = suf.normalize_image(ctx.peak_suppressed,zero_min=True)
#     thinline_inv_cost = suf._normalized_axial_gradient(inv_cost1,vertical_kernel_size=4,dark2bright=True) # This should be moved to config if it's swept ever, but really just sharpens the bottom edge. 
#     for name,inv_cost in [("DP",inv_cost1),("DP_thinline",thinline_inv_cost)]:
#         cost = 1-inv_cost
#         cost = suf.modify_cost_with_ONH_info(cost,ctx.ONH_region,ONH_value_factor=0.5)
#         norm_cost = suf.normalize_image_per_column(cost)
#         Bcs,barrier = suf.calculate_darkness_barrier_and_Bcs(norm_cost,t=t,p=p)
#         DP_path,_ = suf.run_DP_on_cost_matrix(cost,max_step=3,lambda_step=lambda_step,ONH_region=ctx.ONH_region,lambda_step_in_ONH_region=0.001,dbf=dbf,Bcs=Bcs)

#         setattr(ctx,f"{name}_darkness_barrier_img",barrier)
#         setattr(ctx,f"final_{name}_cost",cost)
#         if name == "DP":
#             ctx.ilm_smooth = DP_path
#         else:
#             ctx.ilm_smooth_thinline = DP_path
#     return ctx

def step_ilm_ax_grad_thinner(ctx: ILMContext) -> ILMContext:
    """thin out the otherwise overlapping features"""
    inv_cost = suf.normalize_image(ctx.peak_suppressed,zero_min=True)
    thinline_inv_cost = suf._normalized_axial_gradient(inv_cost,vertical_kernel_size=12,dark2bright=False) # This should be moved to config if it's swept ever, but really just sharpens the bottom edge. 

    ctx.inv_cost = inv_cost
    ctx.thinline_inv_cost = thinline_inv_cost
    return ctx

def step_ilm_DP(ctx: ILMContext) -> ILMContext:
    """2/27/26, to follow after some basic peak suppression, to run the ILM"""
    t = 0.6
    # t = 0.6
    p=1
    dbf=5
    lambda_step=0.01
    
    cost = 1-ctx.thinline_inv_cost
    cost = suf.modify_cost_with_ONH_info(cost,ctx.ONH_region,ONH_value_factor=0.5)
    norm_cost = suf.normalize_image_per_column(cost)
    Bcs,barrier = suf.calculate_darkness_barrier_and_Bcs(norm_cost,t=t,p=p)
    DP_path,_ = suf.run_DP_on_cost_matrix(cost,max_step=3,lambda_step=lambda_step,ONH_region=ctx.ONH_region,lambda_step_in_ONH_region=0.001,dbf=dbf,Bcs=Bcs)

    ctx.final_DP_darkness_barrier_img = barrier
    ctx.final_DP_cost = cost
    ctx.ilm_raw = DP_path
    return ctx

def step_ilm_DP_refiner(ctx: ILMContext) -> ILMContext:
    """should latch back on by selecting the real ILM instead of upshifting. bring back in the pre-thinned peak_suppressed img, which will grab proper region, but guide by the thin-lined result"""
    t = 0.6
    p=1
    dbf=5
    lambda_step=0.01
    
    ctx.DP_refining_tube = suf.apply_gaussian_tube_mul(ctx.inv_cost,guide_y=ctx.ilm_raw,sigma=20,gain=1)
    cost = 1 - ctx.DP_refining_tube
    cost = suf.modify_cost_with_ONH_info(cost,ctx.ONH_region,ONH_value_factor=0.5)
    DP_path,_ = suf.run_DP_on_cost_matrix(cost,max_step=3,lambda_step=lambda_step,ONH_region=ctx.ONH_region,lambda_step_in_ONH_region=0.001)
    ctx.ilm_smooth = DP_path
    return ctx



# def step_ilm_upsample(ctx: ILMContext) -> ILMContext:
#     ilm_raw, ilm_smooth = [
#         suf.upsample_path(
#             e,
#             vertical_factor=ctx.cfg.vertical_factor,
#             original_length=ctx.cfg.original_height,
#         )
#         for e in [ctx.ilm_raw, ctx.ilm_smooth]
#     ]
#     ctx.ilm_raw = ilm_raw
#     ctx.ilm_smooth = ilm_smooth
#     return ctx

def step_ilm_upsample(ctx: ILMContext) -> ILMContext:
    """
    Upsample whichever ILM paths are present on ctx (ilm_raw and/or ilm_smooth).
    If an attribute is missing or None, it's left as-is.
    """
    for attr in ("ilm_raw", "ilm_smooth"):
        v = getattr(ctx, attr, None)
        if v is None:
            continue
        v_up = suf.upsample_path(
            v,
            vertical_factor=ctx.cfg.vertical_factor,
            original_length=ctx.cfg.original_height,
        )
        setattr(ctx, attr, v_up)
    return ctx

def step_ilm_unsmooth(ctx: ILMContext) -> ILMContext:
    """Undo hypersmoothing warp for whatever lines exist (and ILM if present)."""
    names = ["ilm_raw", "ilm_smooth"]

    shift = ctx.hypersmoother_params.hypersmoother_shift_y_full
    for name in names:
        if not hasattr(ctx, name):
            continue
        line = getattr(ctx, name)
        if line is None:
            continue
        # optional logging if your ctx has it
        if hasattr(ctx, "log_history") and callable(getattr(ctx, "log_history")):
            ctx.log_history(f"flat_{name}", line.copy())
        setattr(
            ctx,
            name,
            suf.warp_line_by_shift(line, shift, direction="to_orig"),
        )
    return ctx


def step_ilm_endpoint_plot(ctx: ILMContext) -> ILMContext:
    """do some plotting to summarize the pathway"""
    AB = spu.ArrayBoard(skip=False,plt_display=False,save_tag=f"final_ILM_plot_step_{ctx.ID}")
    AB.add(ctx.original_image,title="original")
    AB.add(ctx.hypersmoother_params.coarse_hypersmoothed_img,title="coarse hypersmoothed image")
    AB.add(ctx.hypersmoothed_img,title="hypersmoothed_img")
    # AB.add(ctx.original_image,lines={'hypersmoothed':ctx.hypersmoother_params.hypersmoother_path},title="original")
    AB.add(ctx.img,title="downsampled and smoothed")
    AB.add(ctx.enh,title="enh_diff")
    # AB.add(ctx.downsampled_img,title="downsampled_img")
    # AB.add(ctx.enh,title="enh_diff (pre suppressed)")
    if hasattr(ctx,'edge_raw') and ctx.edge_raw is not None:
        AB.add(ctx.edge_raw,title="edge raw")
    if hasattr(ctx,'edge') and ctx.edge is not None:
        AB.add(ctx.edge,title="edge")
    # AB.add(ctx.img,lines={'ilm_raw':ctx.ilm_raw},title='downsampled + ilm raw')
    # AB.add(ctx.ilm_tube_cost_DP_path,title='ilm_tube_cost_DP_path')
    # AB.add(ctx.ilm_tube_cost_raw,title='ilm_tube_cost_raw')
    # AB.add(ctx.original_image,lines = {'ilm_raw':ctx.ilm_raw,'ilm_smooth':ctx.ilm_smooth},title='raw and smooth ilms')
    if ctx.inv_cost is not None:
        AB.add(ctx.inv_cost,title='inv_cost')
    if ctx.thinline_inv_cost is not None:
        AB.add(ctx.thinline_inv_cost,title='thinline (real used) inv_cost')
    if ctx.penultimate_DP_cost is not None:
        AB.add(ctx.penultimate_DP_cost,title='penultimate DP cost')
    if ctx.penultimate_DP_darkness_barrier_img is not None:
        AB.add(ctx.penultimate_DP_darkness_barrier_img,title='penultimate_DP_darkness_barrier_img')

    if ctx.DP_refining_tube is not None:
        AB.add(ctx.DP_refining_tube,title='DP_refining_tube')
    if ctx.DP_final_cost is not None:
        AB.add(ctx.DP_final_cost,title='DP_final_cost')

    AB.add(ctx.original_image,lines = {'ilm_raw':ctx.ilm_raw,'ilm_smooth':ctx.ilm_smooth},title='ilms')
    # AB.add(ctx.original_image,lines = {'ilm_smooth':ctx.ilm_smooth,'ilm_smooth_thinline':ctx.ilm_smooth_thinline},title='normal and thinline ilms')
    AB.render()
    # raise Exception("endpoint plotting complete. Terminating here!")
    return ctx
    #ef step_ILM_terminal_plot