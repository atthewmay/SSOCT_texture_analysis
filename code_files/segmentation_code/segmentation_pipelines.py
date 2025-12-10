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




RPE_STEPS_12_6_25: List[ssf.RPEStepFn] = [
    ssf.step_rpe_downsample_and_preprocess,
    ssf.step_rpe_compute_enhancement,
    ssf.step_rpe_peak_suppression,
    # ssf.step_rpe_seed_selection,
    ssf.step_rpe_recalculate_single_seeded_and_reseed,
    # ssf.step_rpe_seed_selection,
    ssf.step_rpe_paths_prob_edge,
    ssf.step_rpe_extract_rpe_raw_and_margin,
    ssf.step_rpe_guided_dp,
    ssf.step_rpe_tube_smoother,
    ssf.step_rpe_smooth_and_upsample,
]

PICKLE_CTX = False
PRODUCTION_MODE = True
def process_bscan_12_6_25(idx_and_img):
    """The new idea is circle back on the peak_suppression stage and identify those cols with only 1 seed point. You take these back to the compute enhancement stage and don't suppress as much"""
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
    rpe_ctx = ssf.run_pipeline(rpe_ctx,steps=RPE_STEPS_12_6_25)
    
    if PICKLE_CTX:
        print("gonna pickle it")
        import pickle
        pickle.dump((ilm_ctx,rpe_ctx),open("/Users/matthewhunt/Research/Iowa_Research/Han_AIR/results/temp_pickle/latest_context_ilm_rpe_pickle",'wb'))
    if PRODUCTION_MODE:
        out_dict = {'rpe_raw':rpe_ctx.rpe_raw,'rpe_smooth':rpe_ctx.rpe_smooth, 'ilm_raw':ilm_ctx.ilm_raw, 'ilm_smooth':ilm_ctx.ilm_smooth} 
        return idx,out_dict
    else:
        return idx, ilm_ctx, rpe_ctx

