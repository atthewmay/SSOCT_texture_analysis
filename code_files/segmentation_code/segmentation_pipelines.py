import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # adds Han_AIR/ to path
import code_files.segmentation_code.segmentation_step_functions as ssf
from typing import List

RPE_STEPS_12_5_25: List[ssf.RPEStepFn] = [
    ssf.step_rpe_downsample_and_preprocess,
    ssf.step_rpe_compute_enhancement,
    ssf.step_rpe_peak_suppression,
    ssf.step_rpe_seed_selection,
    ssf.step_rpe_paths_prob_edge,
    ssf.step_rpe_extract_rpe_raw_and_margin,
    ssf.step_rpe_guided_dp,
    ssf.step_rpe_tube_smoother,
    ssf.step_rpe_smooth_and_upsample,
]

ILM_STEPS_12_5_25 : List[ssf.ILMStepFn] = [
    ssf.step_ilm_downsample_and_preprocess,
    ssf.step_ilm_compute_enhancement,
    ssf.step_ilm_seed_detection,
    ssf.step_ilm_edge_impute,
    ssf.step_ilm_extract_raw,
    ssf.step_ilm_tube_smoother,
    ssf.step_ilm_upsample,
]


def process_bscan_12_5_25(idx_and_img):
    idx, bscan,ONH_info = idx_and_img
    ilm_config = ssf.ILMConfig()
    ilm_ctx = ssf.ILMContext(idx = idx,
                    original_image=bscan.copy(),
                    ONH_region=ONH_info,
                    cfg=ilm_config)
    ilm_ctx = ssf.run_pipeline(ilm_ctx,steps=ILM_STEPS_12_5_25)

    rpe_config = ssf.RPEConfig()
    rpe_ctx = ssf.RPEContext(idx = idx,
                        original_image=bscan.copy(),
                        cfg=rpe_config,
                        ilm_seg=ilm_ctx.ilm_raw//rpe_config.downsample_factor,
                        ONH_region=ONH_info)
    rpe_ctx = ssf.run_pipeline(rpe_ctx,steps=RPE_STEPS_12_5_25)
    return idx, ilm_ctx, rpe_ctx


