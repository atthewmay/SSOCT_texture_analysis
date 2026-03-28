
import pdb
import sys
from pathlib import Path
from token import OP

sys.path.append(str(Path(__file__).resolve().parents[1]))  # adds Han_AIR/ to path
import code_files.segmentation_code.flattening_utility_functions
import code_files.segmentation_code.segmentation_utility_functions as suf
import code_files.segmentation_code.segmentation_plot_utils as spu
import code_files.file_utils as fu
# import code_files.segmentation_utils as su
from dataclasses import dataclass, field,asdict
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

    blur_kernel_size: Optional[int] = 15

    blurred_for_shift: Optional[np.ndarray] = None
    y1_vertical_shifted: Optional[np.ndarray] = None
    y2_vertical_shifted: Optional[np.ndarray] = None
    vertical_shift: Optional[int] = None


# @dataclass
# class twoLayerDPContextChoroidal: # Just a storage for the higher-resolution rpe processing image steps
#     debug_bool: Optional[bool] = True
#     y1: Optional[np.ndarray] = None
#     y2: Optional[np.ndarray] = None
#     img_band: Optional[np.ndarray] = None
#     y1_rescaled: Optional[np.ndarray] = None
#     y2_rescaled: Optional[np.ndarray] = None
#     debug: dict = field(default_factory=dict)

# @dataclass
# class twoLayerDPContextEZ: # Just a storage for the higher-resolution rpe processing image steps
#     input_img: Optional[bool] = True # probably will be top edge of the diff_up_down, obtained via smaller kernel axial grad
#     debug_bool: Optional[bool] = True
#     y1: Optional[np.ndarray] = None
#     y2: Optional[np.ndarray] = None
#     img_band: Optional[np.ndarray] = None
#     y1_rescaled: Optional[np.ndarray] = None
#     y2_rescaled: Optional[np.ndarray] = None
#     debug: dict = field(default_factory=dict)




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
    two_layer_dp_ctx_choroidal: Optional[twoLayerDPContext] = None
    two_layer_dp_ctx_EZ: Optional[twoLayerDPContext] = None
    hypersmoothed_img: np.ndarray = None
    highres_smoothed_img: np.ndarray = None
    hypersmoother_params: HypersmootherParams = field(default_factory=HypersmootherParams)
    img: np.ndarray = None              # working image (may be downsampled)
    downsampled_img: np.ndarray = None              # working image (may be downsampled)
    ilm_seg: np.ndarray = None
    ilm_ctx: ILMContext = None
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
                print(f"Will be loading ckpt from pipeline, found at {p}")
                return pickle.load(f)
        ctx = self.step(ctx)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(ctx, f, protocol=pickle.HIGHEST_PROTOCOL)
        return ctx

def ckpt(step, overwrite=False, save_by_ID=False,cache_file=Path(C['root'])/'results/temp_pickle/pipeline_ctx_cache.pickle',type='RPE'):
    cache_file = cache_file.with_name(f"{cache_file.stem}_{type}{cache_file.suffix}")
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
    ctx.hypersmoothed_img,ctx.hypersmoother_params.hypersmoother_shift_y_full,ctx.hypersmoother_params.hypersmoother_target_y = code_files.segmentation_code.flattening_utility_functions.flatten_to_path(ctx.img,ctx.hypersmoother_params.hypersmoother_path)
    # ctx.hypersmoothed_img,ctx.hypersmoother_params = suf.rpe_hypersmoother_basic_DP(ctx.img)
    # ilm_original = ctx.ilm_seg.copy()
    ctx.ilm_seg_flat = code_files.segmentation_code.flattening_utility_functions.warp_line_by_shift(ctx.ilm_seg,ctx.hypersmoother_params.hypersmoother_shift_y_full,direction="to_flat")
    ctx.ilm_seg = ctx.ilm_seg_flat

    ctx.img = ctx.hypersmoothed_img
    # AB = spu.ArrayBoard()
    # AB.add(ctx.original_image,lines={"ilm_original":ilm_original},title = 'original ilm')
    # AB.add(ctx.hypersmoothed_img,lines={"ilm_warped":ctx.ilm_seg},title = 'warped ilm')
    # AB.render()
    return ctx


from code_files.segmentation_code.custom_dataclasses import *
def step_rpe_hypersmoother_3_7_26(ctx: RPEContext,DEBUG_THIS_FUNCTION=False) -> RPEContext:
    """Run coarse hypersmoother preprocess/DP sweeps and visualize intermediates."""
    from dataclasses import replace
    if DEBUG_THIS_FUNCTION:
        print("going to use debug mode")

        combos = []
        for downsample_factor in [4]:
            for kernel_size in [20]:
                for grad_weight in [0.6]:
                    for blur_sigma in [4]:
                        for suppression_sigma in [12]:
                            for peak_prominence in [0.02]:
                                for rigidity in [20,40]:
                                    for lsonh in [0.0001,0.001,0.01]:
                                        p = HyperPreprocessParams(
                                            ILM_line=None,
                                            grad_weight=grad_weight,
                                            kernel_size=kernel_size,
                                            blur_sigma=blur_sigma,
                                            suppression_sigma=suppression_sigma,
                                            peak_prominence = peak_prominence,
                                        )

                                        combos.append(HyperCombo(
                                            name="",
                                            downsample_factor=downsample_factor,
                                            rigidity=rigidity,
                                            # dp_input_key="combined_peak_suppressed",
                                            preprocess_params=p,
                                            lambda_step_in_onh_region = lsonh,
                                        ))

                            # easy to add later if desired:
                            # combos.append(HyperCombo(
                            #     name="dp_on_grad_shift",
                            #     downsample_factor=downsample_factor,
                            #     rigidity=10,
                            #     dp_input_key="grad_shift",
                            #     preprocess_params=p,
                            # ))
                            #
                            # combos.append(HyperCombo(
                            #     name="dp_on_blurred",
                            #     downsample_factor=downsample_factor,
                            #     rigidity=40,
                            #     dp_input_key="blurred",
                            #     preprocess_params=p,
                            # ))

        def loop_contents(combo: HyperCombo):
            img_copy = ctx.img.copy()
            ds = combo.downsample_factor

            downsampled_img = cv2.resize(
                img_copy,
                (0, 0),
                fx=1.0 / ds,
                fy=1.0 / ds,
            )
            downsampled_img = gaussian_filter(downsampled_img, sigma=blur_sigma) # this was the sneak Tho how does it get set?

            ilm_ds = ctx.ilm_ctx.ilm_smooth[::ds] / ds
            params = replace(combo.preprocess_params, ILM_line=ilm_ds)

            bundle: ImageBundle = suf.HypersmoothPreprocessors._big_blur_grad_combined_peak_suppressed_bundle(
                downsampled_img,
                params,
            )

            # dp_input = getattr(bundle, combo.dp_input_key)
            # dp_inputs = [
            #     ('blurred',bundle.blurred),
            #     ('grad_peak_suppressed',bundle.grad_peak_suppressed),
            #     ('combined_peak_suppressed',bundle.combined_peak_suppressed),
            #     ]
            # # for name,dp_input in dp_inputs:
            # def inner_loop(name,dp_input):
            #     coarse, nonsmooth_y_dp,y_dp, cost = suf.rpe_hypersmoother_DP_3_7_26(
            #         coarse=dp_input,
            #         rigidity=combo.rigidity,
            #         ONH_info=ctx.ONH_region,   # add properly downsampled ONH later if wanted
            #         lambda_step_in_ONH_region=combo.lambda_step_in_onh_region,
            #     )

            #     nonsmooth_hypersmoother_path = suf.upsample_path(
            #         nonsmooth_y_dp,
            #         vertical_factor=ds,
            #         horizontal_factor=ds,
            #     )
 
            #     hypersmoother_path = suf.upsample_path(
            #         y_dp,
            #         vertical_factor=ds,
            #         horizontal_factor=ds,
            #     )
            #     # paths[name] = hypersmoother_path
            #     return name,nonsmooth_hypersmoother_path,hypersmoother_path,cost
            # results = Parallel(n_jobs=3)(delayed(inner_loop)(name,dp_input) for name,dp_input in dp_inputs)

            coarse, nonsmooth_y_dp,y_dp, cost = suf.rpe_hypersmoother_DP_3_7_26(
                coarse=bundle.combined_peak_suppressed,
                rigidity=combo.rigidity,
                ONH_info=ctx.ONH_region,   # add properly downsampled ONH later if wanted
                lambda_step_in_ONH_region=combo.lambda_step_in_onh_region,
            )

            nonsmooth_hypersmoother_path = suf.upsample_path(
                nonsmooth_y_dp,
                vertical_factor=ds,
                horizontal_factor=ds,
            )

            hypersmoother_path = suf.upsample_path(
                y_dp,
                vertical_factor=ds,
                horizontal_factor=ds,
            )
                # paths[name] = hypersmoother_path

            paths = {'hypersmoother_path':hypersmoother_path}
            nonsmooth_paths = {'nonsmooth_hypersmoother_path':nonsmooth_hypersmoother_path}

            ilm_full = suf.upsample_path(
                params.ILM_line,
                vertical_factor=ds,
                horizontal_factor=ds,
            )

            return dict(
                combo=combo,
                params=params,
                downsampled_img=downsampled_img,
                bundle=bundle,
                # dp_input=dp_input,
                # coarse=coarse,
                paths=paths,
                nonsmooth_paths=nonsmooth_paths,
                # hypersmoother_path=hypersmoother_path,
                ilm_full=ilm_full,
                cost=cost,
            )

        results = Parallel(n_jobs=8)(delayed(loop_contents)(c) for c in combos)

        AB = spu.ArrayBoard(skip=False, plt_display=False, save_tag=f"_coarse_hypersmoother_img_{ctx.ID}")
        for r in results:
            combo = r["combo"]
            bundle: ImageBundle = r["bundle"]

            # AB.add(r["downsampled_img"], title=fill(f"downsampled input {combo.name}", 25))
            AB.add(bundle.blurred, title=fill(f"blurred {combo.name}", 25))
            AB.add(bundle.grad_raw, title=fill(f"grad raw {combo.name}", 25))
            # AB.add(bundle.grad_ilm_suppressed, title=fill(f"grad_ilm_suppressed {combo.name}", 25))
            # AB.add(bundle.grad_shift, title=fill(f"grad_shift {combo.name}", 25))
            # AB.add(bundle.grad_peak_suppressed, title=fill(f"grad_peak_suppressed {combo.name}", 25))
            AB.add(bundle.peak_img, title=fill(f"peak_image {combo.name}", 25))
            # AB.add(bundle.combined, title=fill(f"combined {combo.name}", 25))
            AB.add(bundle.combined_peak_suppressed, title=fill(f"combined_peak_suppressed {combo.name}", 25))
            # AB.add(r["dp_input"], title=fill(f"DP input = {combo.dp_input_key}", 25))
            AB.add(r["cost"], title=fill(f"cost {combo.name}", 25))

            AB.add(
                ctx.img,
                lines={
                    **r['nonsmooth_paths'],
                    "ilm": r["ilm_full"],
                },
                title=fill('nonsmooth paths', 25),
            )

            AB.add(
                ctx.img,
                lines={
                    **r['paths'],
                    "ilm": r["ilm_full"],
                },
                title=fill(str(combo), 25),
            )



        AB.render()
        raise Exception("done plotting hypersmoother")
        ctx.hypersmoother_params.hypersmoother_path = results[0]['paths']['combined_peak_suppressed']
    else:

        downsample_factor = 4
        ds = downsample_factor
        kernel_size = 20
        grad_weight = 0.6
        blur_sigma = 4
        suppression_sigma = 12
        peak_prominence = 0.02
        rigidity = 40
        preprocess_params = HyperPreprocessParams(
                ILM_line=None,
                grad_weight=grad_weight,
                kernel_size=kernel_size,
                blur_sigma=blur_sigma,
                suppression_sigma=suppression_sigma,
                peak_prominence = peak_prominence,
            )

        params = HyperCombo(
            name="",
            downsample_factor=downsample_factor,
            rigidity=rigidity,
            preprocess_params=preprocess_params,
        )

        img_copy = ctx.img.copy()
        downsampled_img = cv2.resize(
            img_copy,
            (0, 0),
            fx=1.0 / ds,
            fy=1.0 / ds,
        )
        downsampled_img = gaussian_filter(downsampled_img, sigma=blur_sigma) # this was the sneak

        ilm_ds = ctx.ilm_ctx.ilm_smooth[::ds] / ds
        params.preprocess_params = replace(params.preprocess_params, ILM_line=ilm_ds)

        bundle: ImageBundle = suf.HypersmoothPreprocessors._big_blur_grad_combined_peak_suppressed_bundle(
            downsampled_img,
            params.preprocess_params,
        )

        ctx.hypersmoother_params.coarse_hypersmoothed_img, nonsmooth_y_dp,ctx.hypersmoother_params.hypersmoother_y_dp, cost = suf.rpe_hypersmoother_DP_3_7_26(
            coarse=bundle.combined_peak_suppressed,
            rigidity=params.rigidity,
            ONH_info=ctx.ONH_region,   # add properly downsampled ONH later if wanted
        )

        ctx.hypersmoother_params.hypersmoother_path = suf.upsample_path(
            ctx.hypersmoother_params.hypersmoother_y_dp,
            vertical_factor=ds,
            horizontal_factor=ds,
        )

    ctx.hypersmoothed_img,ctx.hypersmoother_params.hypersmoother_shift_y_full,ctx.hypersmoother_params.hypersmoother_target_y = code_files.segmentation_code.flattening_utility_functions.flatten_to_path(ctx.img,ctx.hypersmoother_params.hypersmoother_path)
    ctx.ilm_seg_flat = code_files.segmentation_code.flattening_utility_functions.warp_line_by_shift(ctx.ilm_seg,ctx.hypersmoother_params.hypersmoother_shift_y_full,direction="to_flat")
    ctx.ilm_seg = ctx.ilm_seg_flat
    # In this debugging pipeline, will just keep the first entry of the loop
    ctx.img = ctx.hypersmoothed_img
    return ctx
 

    # raise Exception("done with plotting")
 
# def step_rpe_hypersmoother_3_7_26(ctx: RPEContext) -> RPEContext:
#     """run the big coarse smoother as inital preprocess. if using this you have to unsmooth at the end, which is why we store the hypersmooth line. We also adjust the ilm line"""


#     combos=[]
#     for downsample_factor in [4]:
#         for vks in [20]:
#             for grad_weight in [0.5,0.8]:
#                 for blur_sigma in [4]:
#                     for suppression_sigma in [12]:
#                         ILM_line = ctx.ilm_ctx.ilm_smooth[::downsample_factor]/downsample_factor
#                         preprocess_kwargs = dict(
#                             ILM_line = ILM_line,
#                             grad_weight = grad_weight,
#                             kernel_size = vks,
#                             blur_sigma = blur_sigma,
#                             suppression_sigma = suppression_sigma,
#                         )
#                         d = dict(
#                             downsample_factor = downsample_factor,
#                             preprocess_kwargs=preprocess_kwargs,
#                                 )
#                         combos.append(d)

#     def loop_contents(combo):
#         img_copy = ctx.img.copy()
#         downsample_factor = combo['downsample_factor']
#         downsampled_img = cv2.resize(
#             img_copy,
#             (0, 0),
#             fx=1.0 / downsample_factor,
#             fy=1.0 / downsample_factor,
#         )
#         blurred_img = gaussian_filter(downsampled_img, sigma=blur_sigma)
#         rigidity=10

#         coarse_hypersmoothed_img1,hypersmoother_path1,_,_ = suf.rpe_hypersmoother_DP(blurred_img, ds_x=1, ds_y=1,
#                                                                                             rigidity=rigidity,
#                                                                                             preprocess_function=suf.HypersmoothPreprocessors._big_blur_grad_ILM_suppressed,
#                                                                                             preprocess_kwargs=combo['preprocess_kwargs'],
#                                                                                             ONH_info=ctx.ONH_region,
#                                                                                             return_cost=True,
#         )

#         coarse_hypersmoothed_img2,hypersmoother_path2,_,_ = suf.rpe_hypersmoother_DP(blurred_img, ds_x=1, ds_y=1,
#                                                                                             rigidity=40,
#                                                                                             preprocess_function=suf.HypersmoothPreprocessors._gblur,
#                                                                                             preprocess_kwargs={'sigma':combo['preprocess_kwargs']['blur_sigma']},
#                                                                                             ONH_info=ctx.ONH_region,
#                                                                                             return_cost=True,
#         )


#         coarse_hypersmoothed_img3,hypersmoother_path3,_,_ = suf.rpe_hypersmoother_DP(blurred_img, ds_x=1, ds_y=1,
#                                                                                             rigidity=rigidity,
#                                                                                             preprocess_function=suf.HypersmoothPreprocessors._big_blur_grad_combined,
#                                                                                             preprocess_kwargs=combo['preprocess_kwargs'],
#                                                                                             ONH_info=ctx.ONH_region,
#                                                                                             return_cost=True,
#         )

#         coarse_hypersmoothed_img4,hypersmoother_path4,_,cost = suf.rpe_hypersmoother_DP(blurred_img, ds_x=1, ds_y=1,
#                                                                                             rigidity=rigidity,
#                                                                                             preprocess_function=suf.HypersmoothPreprocessors._big_blur_grad_combined_peak_suppressed,
#                                                                                             preprocess_kwargs=combo['preprocess_kwargs'],
#                                                                                             ONH_info=ctx.ONH_region,
#                                                                                             return_cost=True,
#         )



#                                                                                             # ONH_info=ctx.ONH_region)
#         hypersmoother_path1 = suf.upsample_path(hypersmoother_path1,vertical_factor=downsample_factor,horizontal_factor=downsample_factor)
#         hypersmoother_path2 = suf.upsample_path(hypersmoother_path2,vertical_factor=downsample_factor,horizontal_factor=downsample_factor)
#         hypersmoother_path3 = suf.upsample_path(hypersmoother_path3,vertical_factor=downsample_factor,horizontal_factor=downsample_factor)
#         hypersmoother_path4 = suf.upsample_path(hypersmoother_path4,vertical_factor=downsample_factor,horizontal_factor=downsample_factor)
#         combo['preprocess_kwargs']['ILM_line'] = suf.upsample_path(combo['preprocess_kwargs']['ILM_line'],vertical_factor=downsample_factor,horizontal_factor=downsample_factor)
#         return coarse_hypersmoothed_img1,hypersmoother_path1, coarse_hypersmoothed_img2,hypersmoother_path2, coarse_hypersmoothed_img3,hypersmoother_path3, coarse_hypersmoothed_img4,hypersmoother_path4,cost, downsampled_img,blurred_img,combo


#     results = Parallel(n_jobs=8)(delayed(loop_contents)(c) for c in combos)


#     AB = spu.ArrayBoard(skip=False,plt_display=False,save_tag=f"_coarse_hypersmoother_img for {ctx.ID}")
#     for r in results:
#         ilm = r[-1]['preprocess_kwargs'].pop("ILM_line")
#         AB.add(r[-2],title=fill(f'blurred_input_img for hypersmoothing for {str(r[-1])}',25))
#         AB.add(r[0],title=f'img for hypersmoothing to perform DP')
#         AB.add(r[2],title=f'img for original-type hypersmoothing to perform DP')
#         AB.add(r[4],title=f'img for combo-type hypersmoothing to perform DP')
#         AB.add(r[6],title=f'img for combo-type peak_suppressed hypersmoothing to perform DP')
#         AB.add(r[8],title=f'sanity checking cost for peak_sup')
#         AB.add(ctx.img,lines={"grad_DP_path":r[1],"original_DP_path":r[3],"product_DP_path":r[5],"product_DP_path":r[7],'ilm':ilm},title=fill(f'hypersmoother_path',width=25))
#     AB.render()
#     raise Exception("done with plotting")
 
#     # """
#     # _,ctx.hypersmoother_params.hypersmoother_path_extras['gradient_line'],_ = suf.rpe_hypersmoother_DP(ctx.img,ds_x=2,ds_y=2,preprocess_function=suf.HypersmoothPreprocessors._gradient,preprocess_kwargs={})
#     #HYPERSMOOTH DEBUGGING
#     """
#     AB = spu.ArrayBoard(skip=False,plt_display=False,save_tag="_coarse_hypersmoother_img")
#     for d_factor in [2,4,8]:
#         for lambda_step in [0,0.05,0.1]:
#             for preprocess_fn,kwargs in ((suf.HypersmoothPreprocessors._gblur,{'sigma':4}),
#                                         (suf.HypersmoothPreprocessors._gradient,{})):
#                 coarse_img,_,y_dp = suf.rpe_hypersmoother_DP(ctx.img,ds_x=d_factor,ds_y=d_factor,lambda_step=lambda_step,preprocess_function=preprocess_fn,preprocess_kwargs=kwargs)
#         # _ = spu.sweep_to_arrayboard(AB,lambda kw: suf.rpe_hypersmoother_basic_DP,base_kwargs={'img':ctx.img},
#         #                         grid = {
#         #                                     'rigidity' : [40],
#         #                                     'sig' : [4.0],
#         #                                     'ds_y' : [4,8],
#         #                                     'ds_x' : [4,8],
#         #                                     'max_step' : [5],
#         #                         })
#                 AB.add(coarse_img,lines={'DP path':y_dp},title=f"d_factor={d_factor},\nlambda = {lambda_step},\nfn={str(preprocess_fn)}")
#     AB.render()

#     raise Exception("Intending to end here")
#     """
#     ctx.hypersmoothed_img,ctx.hypersmoother_params.hypersmoother_shift_y_full,ctx.hypersmoother_params.hypersmoother_target_y = suf.flatten_to_path(ctx.img,ctx.hypersmoother_params.hypersmoother_path)
#     # ctx.hypersmoothed_img,ctx.hypersmoother_params = suf.rpe_hypersmoother_basic_DP(ctx.img)
#     # ilm_original = ctx.ilm_seg.copy()
#     ctx.ilm_seg_flat = suf.warp_line_by_shift(ctx.ilm_seg,ctx.hypersmoother_params.hypersmoother_shift_y_full,direction="to_flat")
#     ctx.ilm_seg = ctx.ilm_seg_flat

#     ctx.img = ctx.hypersmoothed_img
#     # AB = spu.ArrayBoard()
#     # AB.add(ctx.original_image,lines={"ilm_original":ilm_original},title = 'original ilm')
#     # AB.add(ctx.hypersmoothed_img,lines={"ilm_warped":ctx.ilm_seg},title = 'warped ilm')
#     # AB.render()
#     return ctx



def step_rpe_highres_smooth(ctx: RPEContext) -> RPEContext:
    """reflatten any needed images by another line. Output the line, and save the params for later unsmoothing
    Specifically this won't modify the old images bc the highres pathway is totally separate from the original pathway."""

    ctx.highres_smoothed_img,ctx.hypersmoother_params.highres_smoother_shift_y_full,ctx.hypersmoother_params.highres_smoother_target_y = code_files.segmentation_code.flattening_utility_functions.flatten_to_path(ctx.original_image,ctx.rpe_smooth)
    ctx.flat_rpe_smooth = code_files.segmentation_code.flattening_utility_functions.warp_line_by_shift(ctx.rpe_smooth,ctx.hypersmoother_params.highres_smoother_shift_y_full,direction="to_flat")
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
            code_files.segmentation_code.flattening_utility_functions.warp_line_by_shift(line, shift, direction="to_orig"),
        )

    return ctx



    # ctx = helperFunctions._unflatten_lines(ctx,)
    # return ctx
def step_rpe_highres_unsmooth(ctx: RPEContext) -> RPEContext:
    base_lines = [
        ('highres_rpe_raw', 'rpe_raw'),
        ('highres_rpe_refined', 'rpe_refined'),
        ('highres_rpe_refined2', 'rpe_refined2'),
        ('highres_rpe_smooth2', 'rpe_smooth2'),
    ]

    shift = ctx.hypersmoother_params.highres_smoother_shift_y_full

    for hist_name, attr in base_lines:
        line = getattr(ctx.highres_ctx, attr, None)
        if line is not None:
            ctx.log_history(f'flat_{hist_name}', line.copy())
            setattr(
                ctx.highres_ctx,
                attr,
                code_files.segmentation_code.flattening_utility_functions.warp_line_by_shift(line, shift, direction="to_orig")
            )

    two_dp_linenames = ['y1_rescaled', 'y2_rescaled']
    two_dp_types = [ctx.two_layer_dp_ctx, ctx.two_layer_dp_ctx_choroidal, ctx.two_layer_dp_ctx_EZ]

    for dp_instance in two_dp_types:
        if dp_instance is None:
            print(f"unable to find instance of type {type(dp_instance)}")
            continue
        for n in two_dp_linenames:
            line = getattr(dp_instance, n, None)
            if line is not None:
                ctx.log_history(f'flat_{n}', line.copy())
                setattr(dp_instance, n, code_files.segmentation_code.flattening_utility_functions.warp_line_by_shift(line, shift, direction="to_orig"))
            # else:
                # print(f"dp_instance of type: {type(dp_instance)} has None for {n}, yet has dict keys of {dp_instance.__dict__.keys()}")

    return ctx

# def step_rpe_highres_unsmooth(ctx: RPEContext) -> RPEContext:
#     """undoes the hypersmoothing for the lines used in the highres pathway. Includes the rpe_smooth which will have been flattened for this path"""
#     base_lines = [
#         ('highres_rpe_raw',ctx.highres_ctx.rpe_raw),
#         ('highres_rpe_refined',ctx.highres_ctx.rpe_refined),
#         ('highres_rpe_refined2',ctx.highres_ctx.rpe_refined2),
#         ('highres_rpe_smooth2',ctx.highres_ctx.rpe_smooth2),
#         ]

#     # two_dp_names = ['y1', 'y2', 'y1_rescaled', 'y2_rescaled']
#     two_dp_linenames = ['y1_rescaled', 'y2_rescaled']
#     two_dp_types = [ctx.two_layer_dp_ctx,ctx.two_layer_dp_ctx_choroidal,ctx.two_layer_dp_ctx_EZ]
#     # print(ctx.two_layer_dp_ctx)
#     for dp_instance in two_dp_types:
#         for n in two_dp_linenames:
#             if hasattr(dp_instance,n):
#                 # print(f"should be adding {n}")
#                 base_lines.append((n,getattr(dp_instance,n)))

#     # for n in two_dp_linenames:
#     #     if hasattr(ctx.two_layer_dp_ctx,n):
#     #         # print(f"should be adding {n}")
#     #         base_lines.append((n,getattr(ctx.two_layer_dp_ctx,n)))
        

#     for item in base_lines:
#         name,line = item
#         # print(f"logging for flat_{name}")
#         if line is not None:
#             ctx.log_history(f'flat_{name}', line.copy())

#     # unwarped_lines = [suf.warp_line_by_shift(line[1],ctx.hypersmoother_params.highres_smoother_shift_y_full,direction='to_orig') for line in lines] #shoudl refactor
#     shift = ctx.hypersmoother_params.highres_smoother_shift_y_full
#     unwarped = {
#             name: suf.warp_line_by_shift(line, shift, direction="to_orig")
#             for name, line in base_lines if line is not None
#     }

#     # (ctx.highres_ctx.rpe_raw,
#     # ctx.highres_ctx.rpe_refined,
#     # ctx.highres_ctx.rpe_refined2,
#     # ctx.highres_ctx.rpe_smooth2,
#     # ) = unwarped_lines

#     # assign back only the ones that belong to highres_ctx
#     ctx.highres_ctx.rpe_raw      = unwarped.get("highres_rpe_raw",None)
#     ctx.highres_ctx.rpe_refined  = unwarped.get("highres_rpe_refined",None)
#     ctx.highres_ctx.rpe_refined2 = unwarped.get("highres_rpe_refined2",None)
#     ctx.highres_ctx.rpe_smooth2  = unwarped.get("highres_rpe_smooth2",None)

#         # if include_two_layer and getattr(ctx, "two_layer_dp_ctx", None) is not None:
#     for dp_instance in two_dp_types:
#         for n in two_dp_linenames:
#             setattr(dp_instance, n, unwarped[n])

#     return ctx


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
        # suf.upsample_path(
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
        
        unwarped_lines = [code_files.segmentation_code.flattening_utility_functions.warp_line_by_shift(line[1],unflattening_shift,direction='to_orig') for line in lines] #shoudl refactor
        
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
def step_rpe_highres_DP_two_layer(ctx: RPEContext,param_updates=None) -> RPEContext:
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
 

    if param_updates:
        unknown = set(param_updates) - set(base)
        if unknown:
            raise KeyError(f"Unknown params: {unknown}")
        base = base | param_updates

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


def _run_highres_two_layer_choroidal_once(ctx: RPEContext, base: dict):
    import code_files.segmentation_code.two_surface_utils as tsu

    offset_value = np.unique(ctx.flat_rpe_smooth)
    if len(offset_value) != 1:
        print(f"np.unique(ctx.flat_rpe_smooth) is actually {offset_value}")
    offset_value = float(offset_value[0])

    img_band_radius = 30
    r0 = int(max(0, offset_value - img_band_radius))
    r1 = int(min(ctx.highres_ctx.lower_edge_of_tubed.shape[0], offset_value + img_band_radius))

    img_full = ctx.highres_ctx.lower_edge_of_tubed
    img_band = img_full[r0:r1, :]
    cost_int = tsu.make_cost_from_img(img_band, mode="inv_global")

    y1, y2, dbg = tsu.run_two_surface_DP_3_9_26(cost_int, cost_int, return_debug=True, **base)

    def rescale(y):
        return y - img_band_radius + offset_value

    y1_rescaled = rescale(y1)
    y2_rescaled = rescale(y2)

    return dict(
        img_band=img_band,
        y1=y1,
        y2=y2,
        y1_rescaled=y1_rescaled,
        y2_rescaled=y2_rescaled,
        dbg=dbg,
        params=base,
    )


def step_rpe_highres_DP_two_layer_choroidal(ctx: RPEContext, param_updates=None) -> RPEContext:
    base = dict(
        dmin=10, dmax=40,
        max_step1=1, max_step2=1,
        lambda1=0.6, lambda2=0.6,
        prefer_upper_on_single=True,
        prefer_lower_on_single=False,
        single_kappa=0.4,
        kappa_mode='reweight',
        darkness_barrier_factor=0,
        peak_sigma=1,
        peak_distance=5,
        peak_prominance=0.05,
        sep_gamma=0.0,
        depth_a1=0.0,
        depth_a2=0.0,
    )
    if param_updates:
        unknown = set(param_updates) - set(base)
        if unknown:
            raise KeyError(f"Unknown params: {unknown}")
        base = base | param_updates

    out = _run_highres_two_layer_choroidal_once(ctx, base)

    ctx.two_layer_dp_ctx_choroidal.img_band = out["img_band"]
    ctx.two_layer_dp_ctx_choroidal.y1 = out["y1"]
    ctx.two_layer_dp_ctx_choroidal.y2 = out["y2"]
    ctx.two_layer_dp_ctx_choroidal.y1_rescaled = out["y1_rescaled"]
    ctx.two_layer_dp_ctx_choroidal.y2_rescaled = out["y2_rescaled"]
    if ctx.two_layer_dp_ctx_choroidal.debug_bool:
        ctx.two_layer_dp_ctx_choroidal.debug = out["dbg"]

    return ctx

def step_rpe_highres_DP_two_layer_choroidal_sweep_plot(ctx: RPEContext) -> RPEContext:
    AB = spu.ArrayBoard(skip=False, plt_display=False, ncols_max=5, save_tag=f"DP_2layer_sweep_{ctx.ID}")

    AB.add(ctx.original_image, title="original")
    AB.add(ctx.highres_ctx.diff_down_up, title="input (diff_down_up)")

    if getattr(ctx.two_layer_dp_ctx, "img_band", None) is not None:
        AB.add(
            ctx.two_layer_dp_ctx.img_band,
            lines={'y1': ctx.two_layer_dp_ctx.y1, 'y2': ctx.two_layer_dp_ctx.y2},
            title="baseline img band"
        )
        AB.add(
            ctx.highres_smoothed_img,
            lines={'y1': ctx.two_layer_dp_ctx.y1_rescaled, 'y2': ctx.two_layer_dp_ctx.y2_rescaled},
            title="baseline full image"
        )

    base = dict(
        dmin=10, dmax=40,
        max_step1=1, max_step2=1,
        lambda1=0.6, lambda2=0.6,
        prefer_upper_on_single=True,
        prefer_lower_on_single=False,
        single_kappa=0.4,
        kappa_mode='reweight',
        darkness_barrier_factor=0,
        peak_sigma=1,
        peak_distance=5,
        peak_prominance=0.05,
        sep_gamma=0.0,
        depth_a1=0.0,
        depth_a2=0.0,
    )

    param_list = []
    for single_kappa in [0.4, 0.8]:
        for l in [0.6, 1.0]:
            pu = dict(
                single_kappa=single_kappa,
                lambda1=l,
                lambda2=l,
            )
            param_list.append(base | pu)

    def one_run(p):
        return _run_highres_two_layer_choroidal_once(ctx, p)

    results = Parallel(n_jobs=8)(delayed(one_run)(p) for p in param_list)

    ctx.two_layer_dp_ctx.sweep_results = results

    for out in results:
        p = out["params"]
        tag = f"k={p['single_kappa']}, lam={p['lambda1']}"

        AB.add(
            out["img_band"],
            lines={'y1': out["y1"], 'y2': out["y2"]},
            title=f"band {tag}"
        )
        AB.add(
            ctx.highres_smoothed_img,
            lines={'y1': out["y1_rescaled"], 'y2': out["y2_rescaled"]},
            title=f"full {tag}"
        )

    AB.render()
    return ctx

# def step_rpe_highres_DP_two_layer_choroidal(ctx: RPEContext,param_updates=None) -> RPEContext:
#     """
#     specialize in grabbing RPE over choroid by intentially selectign the top line instead
#     """
#     import code_files.segmentation_code.two_surface_utils as tsu

#     AB = spu.ArrayBoard(skip=False,plt_display=False, ncols_max=5, save_tag=f"DP 2-layer step for {ctx.ID}")
#     AB.add(ctx.original_image, title="original")
#     AB.add(ctx.highres_ctx.diff_down_up, title="input (diff_down_up)")
#     AB.add(ctx.two_layer_dp_ctx.img_band, lines={ 'ctx.two_layer_dp_ctx.y1':ctx.two_layer_dp_ctx.y1, 'ctx.two_layer_dp_ctx.y2':ctx.two_layer_dp_ctx.y2}, title="img band from first pass")
#     AB.add(ctx.highres_smoothed_img, lines={'ctx.two_layer_dp_ctx.y1_rescaled':ctx.two_layer_dp_ctx.y1_rescaled, 'ctx.two_layer_dp_ctx.y2_rescaled':ctx.two_layer_dp_ctx.y2_rescaled},
#            title="initial 2 layer")

#     # --- flatten center (your code assumes flat_rpe_smooth is constant)
#     offset_value = np.unique(ctx.flat_rpe_smooth)
#     try:
#         assert len(offset_value) == 1
#     except:
#         print(f"For some reason, the len(offset_value) != 1. Instead, its {len(offset_value)}")
#     offset_value = float(offset_value[0])

#     img_band_radius = 30
#     r0 = int(max(0, offset_value - img_band_radius))
#     r1 = int(min(ctx.highres_ctx.lower_edge_of_tubed.shape[0], offset_value + img_band_radius))

#     img_full = ctx.highres_ctx.lower_edge_of_tubed
#     img_band = img_full[r0:r1, :]

#     # “gradient image as source” (simple vertical abs-grad; replace with your preferred)
#     # dy = np.diff(img_band, axis=0, prepend=img_band[[0], :])
#     # grad_band = np.abs(dy)

#     # Costs: use same pipeline but different sources
#     cost_int = tsu.make_cost_from_img(img_band, mode="inv_colmax")


#     # 2/16 params
    
#     base = dict(
#         dmin=10, dmax=40,
#         max_step1=1, max_step2=1,
#         lambda1=0.6, lambda2=0.6,

#         # new behavior (NMS-based)
#         prefer_upper_on_single=True,
#         prefer_lower_on_single=False,
#         single_kappa = 0.4,          # still the main knob
#         kappa_mode = 'reweight',
#         darkness_barrier_factor = 0,
#         peak_sigma=1,
#         peak_distance= 5,
#         peak_prominance = 0.05, # again if sweep later, refactor into config

#         # optional extras
#         sep_gamma=0.0,
#         depth_a1=0.0,
#         depth_a2=0.0,
#     )
 
#     param_list = []
#     for single_kappa in [0.4,0.8]:
#         for l in [0.6,1]:
#             param_updates = dict(
#                 single_kappa=single_kappa,
#                 lambda1=l,
#                 lambda2=l
#             )
#             param_list.append(param_updates)

#     def rescale(y):  
#         return y - img_band_radius + offset_value
#     def loop_contents(param_updates):
#         local_base = base | param_updates
#         # --- run sweeps on intensity and gradient sources
#         y1, y2, dbg = tsu.run_two_surface_DP_3_9_26(cost_int, cost_int, return_debug=True, **local_base)
#         y1_rescaled,y2_rescaled = rescale(y1),rescale(y2)
#         return y1,y2,dbg,y1_rescaled,y2_rescaled,param_updates

#     results = Parallel(n_jobs=8)(delayed(loop_contents)(pu) for pu in param_list) 

#     for r in results:
#         y1,y2,dbg,y1_rescaled,y2_rescaled,param_updates = r
#         AB.add(img_band, lines={'y1':y1,'y2':y2},title="img_band (intensity)")
#         AB.add(ctx.highres_smoothed_img, lines={'y1':y1_rescaled,'y2':y2_rescaled}, title="choroid performance")

#     AB.render()
#     raise Exception("done plotting")

#     if param_updates:
#         unknown = set(param_updates) - set(base)
#         if unknown:
#             raise KeyError(f"Unknown params: {unknown}")
#         base = base | param_updates

#     # --- run sweeps on intensity and gradient sources
#     y1, y2, dbg = tsu.run_two_surface_DP(cost_int, cost_int, return_debug=True, **base)
#     def rescale(y):  
#         return y - img_band_radius + offset_value
#     y1_rescaled,y2_rescaled = rescale(y1),rescale(y2)
#     # --- visuals (your usual context panels)
#     ctx.two_layer_dp_ctx.img_band = img_band
#     ctx.two_layer_dp_ctx.y1 = y1
#     ctx.two_layer_dp_ctx.y2 = y2
#     ctx.two_layer_dp_ctx.y1_rescaled = y1_rescaled
#     ctx.two_layer_dp_ctx.y2_rescaled = y2_rescaled
#     if ctx.two_layer_dp_ctx.debug_bool:
#         ctx.two_layer_dp_ctx.debug = dbg
#     return ctx

from dataclasses import dataclass
from joblib import Parallel, delayed


@dataclass
class EZPrepSweepConfig:
    down_hblur: int = 40
    up_hblur: int = 50
    down_vertical_kernel_size: int = 25
    up_vertical_kernel_size: int = 15
    down_blur_ksize: int = 15
    up_blur_ksize: int = 15


@dataclass
class EZPrepSweepResult:
    config: Optional[EZPrepSweepConfig] = None
    diff_down_up: Optional[np.ndarray] = None
    diff_up_down: Optional[np.ndarray] = None
    hblur_down: Optional[np.ndarray] = None
    hblur_up: Optional[np.ndarray] = None
    enh_down: Optional[np.ndarray] = None
    enh_up: Optional[np.ndarray] = None
    img_band: Optional[np.ndarray]=None
    gaussian_tubed: Optional[np.ndarray]=None
    tubes: Optional[Any] = None
    tube_edges: Optional[Any] = None


def step_rpe_EZ_egs_two_layer_prep_DEBUG(ctx: RPEContext):

    # configs = []
    # for down_vk in [7, 11, 25]:
    #     for down_hblur in [20, 30, 40]:
    #         for up_hblur in [30, 40, 50]:
    #             configs.append(
    #                 EZPrepSweepConfig(
    #                     down_hblur=down_hblur,
    #                     up_hblur=up_hblur,
    #                     down_vertical_kernel_size=down_vk,
    #                     up_vertical_kernel_size=15,
    #                 )
    #             )

    configs = []
    for down_vk in [25]:
        for up_vertical_kernel_size in [25]:
            for down_hblur in [40]:
                # for up_hblur in [20,30]:
                for up_hblur in [40]:
                    for down_blur_ksize in [20]:
                        for up_blur_ksize in [40]:
                                            # up_params = {'vertical_kernel_size':15,'blur_ksize':40},
                            configs.append(
                                EZPrepSweepConfig(
                                    down_hblur=down_hblur,
                                    up_hblur=up_hblur,
                                    down_vertical_kernel_size=down_vk,
                                    up_vertical_kernel_size=up_vertical_kernel_size,
                                    down_blur_ksize = down_blur_ksize, 
                                    up_blur_ksize = up_blur_ksize, 
                                )
                            )

    offset_value = np.unique(ctx.flat_rpe_smooth)
    if len(offset_value) != 1:
        print(f"np.unique(ctx.flat_rpe_smooth) is {np.unique(ctx.flat_rpe_smooth)}")
    offset_value = float(offset_value[0])



    def loop_contents(config: EZPrepSweepConfig):
        diff_down_up, diff_up_down, hblur_down, hblur_up = suf.diff_boundary_enhance_and_blur_horiz(
            ctx.img,
            down_hblur=config.down_hblur,
            up_hblur=config.up_hblur,
            down_vertical_kernel_size=config.down_vertical_kernel_size,
            up_vertical_kernel_size=config.up_vertical_kernel_size,
        )

        # diff_down_up = suf.shift_grad(diff_down_up,config.up_vertical_kernel_size,shift_direction='down')
        # diff_up_down = suf.shift_grad(diff_up_down,config.up_vertical_kernel_size,shift_direction='up')

        enh_down = suf._boundary_enhance(ctx.img,config.down_vertical_kernel_size,dark2bright=False,blur_ksize=config.down_blur_ksize)
        enh_up = suf._boundary_enhance(ctx.img,config.up_vertical_kernel_size,dark2bright=True,blur_ksize=config.up_blur_ksize)

        tubes = []
        tube_edges = []
        for i,(name,starter) in enumerate([ ('diff_down_up',diff_down_up), ('diff_up_down',diff_up_down) ]):
            gaussian_tubed = suf.apply_gaussian_tube_mul(
                starter,
                ctx.flat_rpe_smooth,
                sigma=ctx.highres_cfg.tube_sigma,
                gain=1,
                hard_window=ctx.highres_cfg.tube_hard_window,
            )
            tubes.append(gaussian_tubed)

            lower_edge_of_tubed = suf._normalized_axial_gradient(
                gaussian_tubed,
                vertical_kernel_size=4,
                dark2bright=True,
            )

            double_lower_edge_of_tubed = suf._normalized_axial_gradient(
                lower_edge_of_tubed,
                vertical_kernel_size=4,
                dark2bright=True,
            )


            upper_edge_of_tubed = suf._normalized_axial_gradient(
                gaussian_tubed,
                vertical_kernel_size=4,
                dark2bright=False,
            )

            double_upper_edge_of_tubed = suf._normalized_axial_gradient(
                upper_edge_of_tubed,
                vertical_kernel_size=4,
                dark2bright=False,
            )
 

            tube_edges.append([lower_edge_of_tubed,upper_edge_of_tubed, double_lower_edge_of_tubed, double_upper_edge_of_tubed])
        # img_band = _make_img_band(lower_edge_of_tubed,offset_value=offset_value)

        # cost_int = tsu.make_cost_from_img(img_band, mode="inv_colmax")

        # y1, y2, dbg = tsu.run_two_surface_DP_3_9_26(cost_int, cost_int, return_debug=True, **base)



        return EZPrepSweepResult(
            config=config,
            diff_down_up=diff_down_up,
            diff_up_down=diff_up_down,
            hblur_down=hblur_down,
            hblur_up=hblur_up,
            gaussian_tubed=None,
            tubes = tubes,
            # img_band=img_band,
            enh_down  = enh_down,
            enh_up  = enh_up,
            tube_edges=tube_edges
        )


    results = Parallel(n_jobs=8)(
        delayed(loop_contents)(config) for config in configs
    )

    img_band_orig = _make_img_band(ctx.highres_ctx.lower_edge_of_tubed,offset_value)

    img_band_radius = 30
    r0 = int(max(0, offset_value - img_band_radius))
    r1 = int(min(ctx.highres_ctx.lower_edge_of_tubed.shape[0], offset_value + img_band_radius))


    AB = spu.ArrayBoard(plt_display=False,save_tag=f"EZ_preprocess_trials for {ctx.ID}")
    AB.add(ctx.highres_smoothed_img  ,lines={'top':ctx.two_layer_dp_ctx.y1_rescaled,'bottom':ctx.two_layer_dp_ctx.y2_rescaled},title='current')
    AB.add(img_band_orig,lines={'top':ctx.two_layer_dp_ctx.y1,'bottom':ctx.two_layer_dp_ctx.y2},title='img_band')



    debugplots=False
    for i, r in enumerate(results):

        if debugplots:
            AB.add(r.enh_down ,  lines={'rpe_smooth':ctx.flat_rpe_smooth},title=fill(f'enh_down'))
            AB.add(r.enh_up ,  lines={'rpe_smooth':ctx.flat_rpe_smooth},title=fill(f'enh_up'))

            AB.add(r.hblur_down,  title=fill(f'hblur_down'))
            AB.add(r.hblur_up,  title=fill(f'hblur_up'))

            AB.add(r.diff_down_up,  title=fill(f'diff_down_up for {r.config}',25))
            AB.add(r.diff_up_down,  title=fill(f'diff_up_down'))

        # AB.add_array(r.gaussian_tubed, row=i, col=1, title="gaussian_tubed")
        # AB.add(r.lower_edge_of_tubed, title="lower_edge")
        # AB.add(r.upper_edge_of_tubed, title="upper_edge")



        img_band_lower = _make_img_band(r.tube_edges[0][0],center_value=offset_value)
        AB.add(img_band_lower, title=fill(f"img_band_lower for diff_down_up {r.config}",25))

        img_band_lower = _make_img_band(r.tube_edges[0][2],center_value=offset_value)
        AB.add(img_band_lower, title=fill(f"img_band_lower for double_diff_down_up {r.config}",25))

        img_band_full = _make_img_band(r.tubes[0],center_value=offset_value)
        AB.add(img_band_full, title=fill(f"img band for for diff_down_up",25))
        img_band_full = _make_img_band(r.tubes[1],center_value=offset_value)
        AB.add(img_band_full, title=fill(f"img band for for diff_up_down",25))

        img_band_upper = _make_img_band(r.tube_edges[1][1],center_value=offset_value)
        AB.add(img_band_upper, title=fill(f"img_band_upper for diff_up_down {r.config}",25))

        img_band_upper = _make_img_band(r.tube_edges[1][3],center_value=offset_value)
        AB.add(img_band_upper, title=fill(f"img_band_upper for diff_up_down {r.config}",25))

        # img_band_lower2 = _make_img_band(r.tube_edges[1][0],offset_value=offset_value)
        # AB.add(img_band_lower2 , title=fill(f"img_band_lower for diff_up_down",25))

        # names = ['diff_down_up','diff_up_down']
        # img_band_upper = _make_img_band(te[1],offset_value=offset_value)
        # AB.add(img_band_upper, title=f"img_band_upper for {name}")
        # for name,te in zip(names,r.tube_edges):
        #     img_band_lower = _make_img_band(te[0],offset_value=offset_value)
        #     AB.add(img_band_lower, title=f"img_band_lower for {name} {r.config}")
        #     img_band_upper = _make_img_band(te[1],offset_value=offset_value)
        #     AB.add(img_band_upper, title=f"img_band_upper for {name}")

    AB.render()

    raise Exception('done')

  
def _rpe_EZ_egs_two_layer_prep(ctx: RPEContext,num_gradients=1):
    """distilled from the above. Option to repeat gradients (better separate choroidal junk)"""
    _, diff_up_down, _, _ = suf.diff_boundary_enhance_and_blur_horiz(
        ctx.img,
        down_hblur=40,
        up_hblur=40,
        down_vertical_kernel_size=25,
        up_vertical_kernel_size=25,
    )


    diff_up_down_tubed = suf.apply_gaussian_tube_mul(
        diff_up_down,
        ctx.flat_rpe_smooth,
        sigma=ctx.highres_cfg.tube_sigma,
        gain=1,
        hard_window=ctx.highres_cfg.tube_hard_window,
    )
    grad_img = diff_up_down_tubed

    for i in range(num_gradients):
        grad_img = suf._normalized_axial_gradient(
            grad_img,
            vertical_kernel_size=4,
            dark2bright=False,
        )
    input_img = grad_img
    return input_img # may add more debug output latyer



def _make_img_band(img_to_band,center_value:float,up_band_radius=30,down_band_radius=30):
    r0 = int(max(0, center_value - up_band_radius))
    r1 = int(min(img_to_band.shape[0], center_value + down_band_radius))
    img_band = img_to_band[r0:r1, :]
    return img_band

def _run_highres_two_layer_EZ_once(ctx: RPEContext, base: dict):
    import code_files.segmentation_code.two_surface_utils as tsu
    input_img = _rpe_EZ_egs_two_layer_prep(ctx,base.pop('num_gradients',1))

    band_center_value = np.unique(ctx.flat_rpe_smooth)
    if len(band_center_value) != 1:
        print(f"np.unique(ctx.flat_rpe_smooth) is {np.unique(ctx.flat_rpe_smooth)}")
    band_center_value = float(band_center_value[0])

    up_band_radius = 20
    down_band_radius = 40
    img_band = _make_img_band(input_img,band_center_value,up_band_radius=up_band_radius,down_band_radius=down_band_radius)
    cost_int = tsu.make_cost_from_img(img_band, mode=base.pop('cost_mode'))

    t = base.pop("darkness_barrier_t",None)
    p = base.pop("darkness_barrier_p",None)
    if base['darkness_barrier_factor'] != 0:
        norm_cost = suf.normalize_image(cost_int,zero_min=True)
        # assert norm_cost == cost_int
        Bcs,barrier = suf.calculate_darkness_barrier_and_Bcs(norm_cost,t=t,p=p)

        # sloppy to be sure!!
        base['darkness_Bcs'] = Bcs
        base['darkness_barrier'] = barrier

    y1, y2, dbg = tsu.run_two_surface_DP_3_9_26(cost_int, cost_int, return_debug=True, **base)
    if dbg is not None:
        dbg['cost'] = cost_int
    if base.get('darkness_barrier') is not None:
        dbg['darkness_barrier'] = base.get('darkness_barrier')

    def rescale(y):
        return y - up_band_radius + band_center_value

    y1_rescaled = rescale(y1)
    y2_rescaled = rescale(y2)

    return dict(
        img_band=img_band,
        y1=y1,
        y2=y2,
        y1_rescaled=y1_rescaled,
        y2_rescaled=y2_rescaled,
        dbg=dbg,
        params=base,
    )

def step_rpe_highres_DP_two_layer_EZ_DEBUG(ctx: RPEContext, param_updates=None) -> RPEContext:
    base = dict(
        dmin=10, dmax=40,
        max_step1=1, max_step2=1,
        lambda1=0.6, lambda2=0.6,
        prefer_upper_on_single=False,
        prefer_lower_on_single=True,
        single_kappa=0.4,
        kappa_mode='reweight',
        darkness_barrier_factor=0.5,
        darkness_barrier_t=None,
        darkness_barrier_p=None,
        peak_sigma=1,
        peak_distance=5,
        peak_prominance=0.05,
        sep_gamma=0.0,
        depth_a1=0.0,
        depth_a2=0.0,
        num_gradients=1,
        cost_mode = "inv_colmax"
    )

    param_update_list = []
    for darkness_barrier_factor in [0,0.5]:
        for darkness_barrier_t in [0.6]:
            for darkness_barrier_p in [1]:
                for num_gradients in [1]:
                    # for cost_mode in ['inv_colmax','inv_global']:
                    for cost_mode in ['inv_global']:
                        param_update = dict(
                            darkness_barrier_factor=darkness_barrier_factor,
                            darkness_barrier_t =darkness_barrier_t,
                            darkness_barrier_p =darkness_barrier_p,
                            num_gradients =num_gradients,
                            cost_mode =cost_mode,
                        )
                        param_update_list.append(param_update)

    def loop_contents(param_update,base_dict):
        if param_update:
            unknown = set(param_update) - set(base_dict)
            if unknown:
                raise KeyError(f"Unknown params: {unknown}")
            base_dict = base_dict | param_update
        

        out = _run_highres_two_layer_EZ_once(ctx, base_dict)
        out['param_update'] = param_update
        return out

    # r = loop_contents(param_update_list[0],base.copy())
    results = Parallel(n_jobs=8)(delayed(loop_contents)(param_update,base.copy()) for param_update in param_update_list)

    AB = spu.ArrayBoard(plt_display=False,save_tag=f"EZ_DP sweep for {ctx.ID}")
    AB.add(ctx.highres_smoothed_img  ,lines={'top':ctx.two_layer_dp_ctx.y1_rescaled,'bottom':ctx.two_layer_dp_ctx.y2_rescaled},title='current')
    for out in results:
        AB.add(out['dbg']['cost'],title=fill(f"cost_img for EZ version {spu.filter_small_titles_for_dict(out['param_update'])}",25))
        if out['dbg'].get('darkness_barrier') is not None:
            AB.add(out['dbg']['darkness_barrier'],title='barrier_img for EZ version')
        if out['dbg'].get('peaks') is not None:
            peak_img = spu.overlay_peaks_on_image(1-out['dbg']['cost'],out['dbg']['peaks'])
            AB.add(peak_img,title='peak_img')
        else:
            print("skipping peak img")
        AB.add(out['img_band'],lines={'top':out['y1'],'bottom':out['y2']},title='img_band')
    AB.render()
    raise Exception("done!")

def step_rpe_highres_DP_two_layer_EZ(ctx: RPEContext) -> RPEContext:
    base = dict(
        dmin=10, dmax=40,
        max_step1=1, max_step2=1,
        lambda1=0.6, lambda2=0.6,
        prefer_upper_on_single=False,
        prefer_lower_on_single=True,
        single_kappa=0.4,
        kappa_mode='reweight',

        darkness_barrier_factor=0,
        darkness_barrier_t=None,
        darkness_barrier_p=None,

        peak_sigma=1,
        peak_distance=5,
        peak_prominance=0.05,

        sep_gamma=0.0,
        depth_a1=0.0,
        depth_a2=0.0,

        num_gradients=1,
        cost_mode="inv_global",
    )
    if base['darkness_barrier_factor'] != 0:
        assert base['darkness_barrier_t'] is not None

    out = _run_highres_two_layer_EZ_once(ctx, base)

    ctx.two_layer_dp_ctx_EZ.img_band = out["img_band"]
    ctx.two_layer_dp_ctx_EZ.y1 = out["y1"]
    ctx.two_layer_dp_ctx_EZ.y2 = out["y2"]
    ctx.two_layer_dp_ctx_EZ.y1_rescaled = out["y1_rescaled"]
    ctx.two_layer_dp_ctx_EZ.y2_rescaled = out["y2_rescaled"]
    if ctx.two_layer_dp_ctx_EZ.debug_bool:
        ctx.two_layer_dp_ctx_EZ.debug = out["dbg"]
    return ctx


def step_rpe_vertical_shift_refine(ctx: RPEContext) -> RPEContext:
    """Rigid vertical-shift refinement of two-layer DP outputs.

    - EZ ctx: refine y1, then shift y2 by the same amount
    - original/choroid ctxs: refine y2, then shift y1 by the same amount
    - stores results as new attrs on each sub-ctx:
        y1_vertical_shifted
        y2_vertical_shifted
        vertical_shift
    """
    img = ctx.original_image

    targets = [
        (ctx.two_layer_dp_ctx_EZ, 'y2_rescaled',[4,1]),
        (ctx.two_layer_dp_ctx_choroidal, 'y1_rescaled',[1,4]),
        (ctx.two_layer_dp_ctx, 'y2_rescaled',[1,4]),
    ]

    for dp_ctx, line_name_to_refine,up_down_range in targets:
        if dp_ctx is None:
            continue
        y1_rescaled = np.asarray(dp_ctx.y1_rescaled).astype(np.int32)
        y2_rescaled = np.asarray(dp_ctx.y2_rescaled).astype(np.int32)

        line = y1_rescaled if line_name_to_refine == "y1_rescaled" else y2_rescaled

        refined_line, shift, _, blurred = suf.refine_line_by_brightness(
            original_image=img,
            blur_kernel_size=dp_ctx.blur_kernel_size,
            proposed_line=line,
            up_range=up_down_range[0],
            down_range=up_down_range[1],
        )

        if line_name_to_refine == 'y1_rescaled':
            dp_ctx.y1_vertical_shifted = refined_line
            dp_ctx.y2_vertical_shifted = y2_rescaled + shift
        else:
            dp_ctx.y2_vertical_shifted = refined_line
            dp_ctx.y1_vertical_shifted = y1_rescaled + shift

        dp_ctx.vertical_shift = shift
        dp_ctx.blurred_for_shift = blurred

    return ctx

def step_rpe_vertical_shift_refine_DEBUG(ctx: RPEContext) -> RPEContext:
    """Debug plotting for rigid vertical-shift refinement.

    Shows only the RPE proposal line from each method:
    - original: y2_rescaled
    - choroidal: y2_rescaled
    - EZ: y1_rescaled

    First panel: original unshifted proposal lines together
    Subsequent panels: shifted proposal lines together for each blur kernel
    """
    img = ctx.original_image if hasattr(ctx, "original_image") and ctx.original_image is not None else ctx.img

    ctx_tags = ["EZ", "choroidal", "original"]
    blur_kernel_sizes = [5, 9, 15]

    combos = []
    for blur_kernel_size in blur_kernel_sizes:
        combos.append(dict(
            blur_kernel_size=blur_kernel_size,
        ))

    # -------- precompute original proposal lines once --------
    base_line_dict = {}

    if ctx.two_layer_dp_ctx_EZ is not None:
        base_line_dict["EZ_RPE_seg"] = np.asarray(ctx.two_layer_dp_ctx_EZ.y2_rescaled).astype(np.int32)

    if ctx.two_layer_dp_ctx_choroidal is not None:
        base_line_dict["choroidal_RPE_seg"] = np.asarray(ctx.two_layer_dp_ctx_choroidal.y1_rescaled).astype(np.int32)

    if ctx.two_layer_dp_ctx is not None:
        base_line_dict["original_RPE_seg"] = np.asarray(ctx.two_layer_dp_ctx.y2_rescaled).astype(np.int32)

    img_flat_base, base_line_dict_flat = spu.flatten_for_plotting(
        img,
        line_dict=base_line_dict,
        flattener=ctx.rpe_smooth,
    )

    img_narrow_base, base_line_dict_narrow = spu.narrow_img_by_line(
        img_flat_base,
        base_line_dict_flat,
        adjust_by="original_RPE_seg" if "original_RPE_seg" in base_line_dict_flat else list(base_line_dict_flat.keys())[0],
        top_pad=120,
        bottom_pad=120,
    )

    def loop_contents(combo):
        blur_kernel_size = combo["blur_kernel_size"]

        shifted_line_dict = {}
        shift_dict = {}
        score_dict = {}

        blurred = None

        for which_ctx in ctx_tags:
            if which_ctx == "EZ":
                dp_ctx = ctx.two_layer_dp_ctx_EZ
                line_name_to_refine = "y2_rescaled"
                up_range, down_range = 5, 10
                output_name = "EZ_RPE_seg_shifted"
            elif which_ctx == "choroidal":
                dp_ctx = ctx.two_layer_dp_ctx_choroidal
                line_name_to_refine = "y1_rescaled"
                up_range, down_range = 10, 5
                output_name = "choroidal_RPE_seg_shifted"
            else:
                dp_ctx = ctx.two_layer_dp_ctx
                line_name_to_refine = "y2_rescaled"
                up_range, down_range = 10, 5
                output_name = "original_RPE_seg_shifted"

            if dp_ctx is None:
                continue

            y1_rescaled = np.asarray(dp_ctx.y1_rescaled).astype(np.int32)
            y2_rescaled = np.asarray(dp_ctx.y2_rescaled).astype(np.int32)

            line = y1_rescaled if line_name_to_refine == "y1_rescaled" else y2_rescaled

            refined_line, shift, score, blurred = suf.refine_line_by_brightness(
                original_image=img,
                blur_kernel_size=blur_kernel_size,
                proposed_line=line,
                up_range=up_range,
                down_range=down_range,
            )

            shifted_line_dict[output_name] = refined_line
            shift_dict[which_ctx] = shift
            score_dict[which_ctx] = score

        img_flat, shifted_line_dict_flat = spu.flatten_for_plotting(
            img,
            line_dict=shifted_line_dict,
            flattener=ctx.rpe_smooth,
        )

        blurred_flat, _ = spu.flatten_for_plotting(
            blurred,
            line_dict={},
            flattener=ctx.rpe_smooth,
        )

        adjust_by = "original_RPE_seg_shifted" if "original_RPE_seg_shifted" in shifted_line_dict_flat else list(shifted_line_dict_flat.keys())[0]

        img_narrow, shifted_line_dict_narrow = spu.narrow_img_by_line(
            img_flat,
            shifted_line_dict_flat,
            adjust_by=adjust_by,
            top_pad=120,
            bottom_pad=120,
        )

        blurred_narrow, shifted_line_dict_blurred_narrow = spu.narrow_img_by_line(
            blurred_flat,
            shifted_line_dict_flat,
            adjust_by=adjust_by,
            top_pad=120,
            bottom_pad=120,
        )

        return dict(
            combo=combo,
            img_narrow=img_narrow,
            blurred_narrow=blurred_narrow,
            shifted_line_dict_narrow=shifted_line_dict_narrow,
            shifted_line_dict_blurred_narrow=shifted_line_dict_blurred_narrow,
            shift_dict=shift_dict,
            score_dict=score_dict,
        )

    results = Parallel(n_jobs=8)(delayed(loop_contents)(c) for c in combos)

    AB = spu.ArrayBoard(skip=False, plt_display=False, save_tag=f"_vertical_shift_refine_{ctx.ID}")

    AB.add(
        img,
        lines=base_line_dict,
        title="Original with non-refined RPE-proposal lines"
    )
    AB.add(
        img_flat_base,
        lines=base_line_dict_flat,
        title="Original flat with non-refined RPE-proposal lines"
    )

    AB.add(
        img_narrow_base,
        lines=base_line_dict_narrow,
        title="Original with non-refined RPE-proposal lines"
    )

    for r in results:
        if r is None:
            continue

        k = r["combo"]["blur_kernel_size"]
        shift_txt = ", ".join([f"{k2}:{v}" for k2, v in r["shift_dict"].items()])

        # AB.add(
        #     r["img_narrow"],
        #     lines=r["shifted_line_dict_narrow"],
        #     title=fill(f"shifted proposals k={k} ({shift_txt})", 25),
        # )

        AB.add(
            r["blurred_narrow"],
            lines=r["shifted_line_dict_blurred_narrow"],
            title=fill(f"blurred k={k} ({shift_txt})", 25),
        )

    AB.render()
    raise Exception("done plotting vertical shift refine")


def step_rpe_endpoint_plot(ctx: RPEContext) -> RPEContext:
    """do some plotting to summarize the pathway"""
    print("will be trying to plot the RPE now")
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
        AB.add(ctx.enh,title="enh (pre suppressed)") # AS of 2_28 pipeline, don't use the enh_diff nitially
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

def step_rpe_choroidal_EZ_endpoint_plot(ctx: RPEContext) -> RPEContext:
    """do some plotting to summarize the pathway"""
    print("will be trying to plot the RPE now")
    AB = spu.ArrayBoard(skip=False,plt_display=False,
                        temp_fig_location=f"/Users/matthewhunt/Research/Iowa_Research/Han_AIR/results/temp_choroidal_grab_figs/final_EZ_CHR_plot_step_{ctx.ID}")


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
        AB.add(ctx.enh,title="enh (pre suppressed)") # AS of 2_28 pipeline, don't use the enh_diff nitially
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
    AB.add(ctx.highres_ctx.diff_down_up,title="highfreq_diff_down_up")
    AB.add(ctx.highres_ctx.lower_edge_of_tubed,title="lower_edge_of_tubed")
    # AB.add(ctx.highres_ctx.highfreq_diff_down_up,title="highest-res gradient")
    # Now add the two-layer stuff
    if ctx.two_layer_dp_ctx.debug:
        print("adding debug info")
        dbg = ctx.two_layer_dp_ctx.debug
        peaks = dbg['peaks']
        peak_img = spu.overlay_peaks_on_image(ctx.two_layer_dp_ctx.img_band,peaks)
        AB.add(peak_img,  title='original 2-layer DP peak img')
        # AB.add(ctx.two_layer_dp_ctx.debug,lines = {'y1':ctx.two_layer_dp_ctx.y1,'y2': ctx.two_layer_dp_ctx.y2},title='img_band')
    AB.add(ctx.two_layer_dp_ctx.img_band,lines = {'y1':ctx.two_layer_dp_ctx.y1,'y2': ctx.two_layer_dp_ctx.y2},title='original 2-layer')
    # AB.add(ctx.two_layer_dp_ctx.img_band,lines = {'y1':ctx.two_layer_dp_ctx.y1,'y2': ctx.two_layer_dp_ctx.y2},title='original 2-layer')

    if ctx.two_layer_dp_ctx_choroidal.debug:
        print("adding debug info")
        dbg = ctx.two_layer_dp_ctx_choroidal.debug
        peaks = dbg['peaks']
        peak_img = spu.overlay_peaks_on_image(ctx.two_layer_dp_ctx_choroidal.img_band,peaks)
        AB.add(peak_img,  title='choroidal 2-layer DP peak img')
        # AB.add(ctx.two_layer_dp_ctx_choroidal.debug,lines = {'y1':ctx.two_layer_dp_ctx_choroidal.y1,'y2': ctx.two_layer_dp_ctx_choroidal.y2},title='img_band')
    AB.add(ctx.two_layer_dp_ctx_choroidal.img_band,lines = {'y1':ctx.two_layer_dp_ctx_choroidal.y1,'y2': ctx.two_layer_dp_ctx_choroidal.y2},title='choroidal 2-layer')
    # AB.add(ctx.two_layer_dp_ctx_choroidal.img_band,lines = {'y1':ctx.two_layer_dp_ctx_choroidal.y1,'y2': ctx.two_layer_dp_ctx_choroidal.y2},title='original 2-layer')

    if ctx.two_layer_dp_ctx_EZ is not None:
        if ctx.two_layer_dp_ctx_choroidal.debug:
            print("adding debug info")
            dbg = ctx.two_layer_dp_ctx_EZ.debug
            peaks = dbg['peaks']
            peak_img = spu.overlay_peaks_on_image(ctx.two_layer_dp_ctx_EZ.img_band,peaks)
            AB.add(peak_img,  title='EZ 2-layer DP peak img')
            # AB.add(ctx.two_layer_dp_ctx_EZ.debug,lines = {'y1':ctx.two_layer_dp_ctx_EZ.y1,'y2': ctx.two_layer_dp_ctx_EZ.y2},title='img_band')
        AB.add(ctx.two_layer_dp_ctx_EZ.img_band,lines = {'y1':ctx.two_layer_dp_ctx_EZ.y1,'y2': ctx.two_layer_dp_ctx_EZ.y2},title='EZ 2-layer')
        # AB.add(ctx.two_layer_dp_ctx_EZ.img_band,lines = {'y1':ctx.two_layer_dp_ctx_EZ.y1,'y2': ctx.two_layer_dp_ctx_EZ.y2},title='original 2-layer')

    preshift_lines = spu.flatNarrowPlotterLines(
        original_method_RPE = ctx.two_layer_dp_ctx.y2_rescaled,
        choroidal_method_RPE = ctx.two_layer_dp_ctx_choroidal.y1_rescaled,
        EZ_method_RPE = ctx.two_layer_dp_ctx_EZ.y2_rescaled,
    )
    flat_narrow_img,flat_narrow_lines = spu.prep_flat_narrow_plot(ctx.original_image,preshift_lines,smoother=ctx.rpe_smooth,pad_amt=100)
    AB.add(flat_narrow_img,lines = flat_narrow_lines,title = 'pre-shifted lines')

    postshift_lines = spu.flatNarrowPlotterLines(
        original_method_RPE = ctx.two_layer_dp_ctx.y2_vertical_shifted,
        choroidal_method_RPE = ctx.two_layer_dp_ctx_choroidal.y1_vertical_shifted,
        EZ_method_RPE = ctx.two_layer_dp_ctx_EZ.y2_vertical_shifted,
    )
    flat_narrow_img,flat_narrow_lines = spu.prep_flat_narrow_plot(ctx.two_layer_dp_ctx.blurred_for_shift,postshift_lines,smoother=ctx.rpe_smooth,pad_amt=100)
    vert_shifts = [(name,dp_ctx.vertical_shift) for name,dp_ctx in 
                   dict( original_method=ctx.two_layer_dp_ctx,
                    choroidal_method=ctx.two_layer_dp_ctx_choroidal,
                    EZ_method=ctx.two_layer_dp_ctx_EZ).items()]
    shift_txt = ", ".join([f"{n}:{v}" for n, v in vert_shifts])
    AB.add(flat_narrow_img,lines = flat_narrow_lines,title = fill(shift_txt,25))


    # Summarizing original imgs
    AB.add(ctx.original_image,lines = { 
                                            "rpe_smooth":ctx.rpe_smooth,
                                        "two_layer_y1":ctx.two_layer_dp_ctx.y1_vertical_shifted,
                                        "two_layer_y2":ctx.two_layer_dp_ctx.y2_vertical_shifted,
                                        },title="Original with two-layer lines")

    # Summarizing original imgs
    AB.add(ctx.original_image,lines = { 
                                        "two_layer_y1":ctx.two_layer_dp_ctx_choroidal.y1_vertical_shifted,
                                        "two_layer_y2":ctx.two_layer_dp_ctx_choroidal.y2_vertical_shifted,
                                        },title="Choroidal with two-layer lines")

    AB.add(ctx.original_image,lines = { 
                                        "two_layer_y1":ctx.two_layer_dp_ctx_EZ.y1_vertical_shifted,
                                        "two_layer_y2":ctx.two_layer_dp_ctx_EZ.y2_vertical_shifted,
                                        },title="EZ with two-layer lines")

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
    # assert len(value)==1
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
    # assert len(value)==1
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
    # assert len(offset_value) == 1
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
    ctx.hypersmoothed_img,ctx.hypersmoother_params.hypersmoother_shift_y_full,ctx.hypersmoother_params.hypersmoother_target_y = code_files.segmentation_code.flattening_utility_functions.flatten_to_path(ctx.original_image,ctx.hypersmoother_params.hypersmoother_path)
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
            code_files.segmentation_code.flattening_utility_functions.warp_line_by_shift(line, shift, direction="to_orig"),
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