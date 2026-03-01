import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # adds Han_AIR/ to path
import code_files.segmentation_code.segmentation_step_functions as ssf
from typing import List

# RPE_STEPS_12_5_25: List[ssf.RPEStepFn] = [
#     ssf.step_rpe_downsample_and_preprocess,
#     ssf.step_rpe_compute_enhancement,
#     ssf.step_rpe_peak_suppression,
#     ssf.step_rpe_seed_selection,
#     ssf.step_rpe_paths_prob_edge,
#     ssf.step_rpe_extract_rpe_raw_and_margin,
#     ssf.step_rpe_guided_dp,
#     ssf.step_rpe_tube_smoother,
#     ssf.step_rpe_upsample,
# ]

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
    ssf.step_rpe_upsample,
]

PICKLE_CTX = False
# PRODUCTION_MODE = True
def process_bscan_12_6_25(idx_and_img,production_mode,rpe_seg_steps):
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
    rpe_ctx = ssf.run_pipeline(rpe_ctx,steps=rpe_seg_steps)
    
    if PICKLE_CTX:
        print("gonna pickle it")
        import pickle
        pickle.dump((ilm_ctx,rpe_ctx),open("/Users/matthewhunt/Research/Iowa_Research/Han_AIR/results/temp_pickle/latest_context_ilm_rpe_pickle",'wb'))
    if production_mode:
        out_dict = {'rpe_raw':rpe_ctx.rpe_raw,'rpe_smooth':rpe_ctx.rpe_smooth, 'ilm_raw':ilm_ctx.ilm_raw, 'ilm_smooth':ilm_ctx.ilm_smooth} 
        return idx,out_dict
    else:
        return idx, ilm_ctx, rpe_ctx


RPE_STEPS_1_3_26: List[ssf.RPEStepFn] = [
    ssf.step_rpe_init_working,
    ssf.step_rpe_hypersmoother,
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
    # ssf.ckpt(ssf.step_rpe_smooth_and_upsample,overwrite=False), # Checkpoint here
    ssf.ckpt(ssf.step_rpe_upsample,overwrite=False,save_by_ID=True),
    # ssf.ckpt(ssf.step_rpe_smooth_and_upsample,overwrite=True),# Checkpoint here
    ssf.step_rpe_highres_flat_guided,
    # ssf.step_rpe_unsmooth,
]

# RPE_STEPS_1_3_26 = ssf.filter_pipeline(RPE_STEPS_1_3_26)
# print(f"RPE_STEPS_1_3_26 is equal to {RPE_STEPS_1_3_26}")


def process_bscan_1_3_26(idx_and_img,production_mode,rpe_seg_steps,ilm_seg_steps=ILM_STEPS_12_5_25):
    """The only change is I'm passing in the full res ilm_raw and downsampling it later during the appropriate downsmpling step (y only for now)"""
    idx, bscan,ONH_info,work_id = idx_and_img
    ilm_config = ssf.ILMConfig()
    ilm_ctx = ssf.ILMContext(idx = idx,
                                ID=work_id,
                    original_image=bscan.copy(),
                    ONH_region=ONH_info,
                    cfg=ilm_config)
    ilm_ctx = ssf.run_pipeline(ilm_ctx,steps=ilm_seg_steps)

    rpe_config = ssf.RPEConfig()
    highres_rpe_config = ssf.HighResConfig()
    highres_rpe_context = ssf.HighResContext()
    two_layer_dp_context  = ssf.twoLayerDPContext()
    rpe_ctx = ssf.RPEContext(idx = idx, # Idx here is the work index. That should be a valid way to tag the saved ckpt
                        ID = work_id,
                        original_image=bscan.copy(),
                        cfg=rpe_config,
                        highres_cfg=highres_rpe_config,
                        highres_ctx=highres_rpe_context,
                        two_layer_dp_ctx=two_layer_dp_context,
                        ilm_seg=ilm_ctx.ilm_raw,
                        ONH_region=ONH_info)
    rpe_ctx = ssf.run_pipeline(rpe_ctx,steps=rpe_seg_steps)
    # print(rpe_ctx.highres_cfg)
    
    if PICKLE_CTX:
        print("gonna pickle it")
        import pickle
        pickle.dump((ilm_ctx,rpe_ctx),open("/Users/matthewhunt/Research/Iowa_Research/Han_AIR/results/temp_pickle/latest_context_ilm_rpe_pickle",'wb'))
    if production_mode:
        out_dict = {'rpe_raw':rpe_ctx.rpe_raw,'rpe_smooth':rpe_ctx.rpe_smooth, 'ilm_raw':ilm_ctx.ilm_raw, 'ilm_smooth':ilm_ctx.ilm_smooth} 
        return idx,out_dict
    else:
        return idx, ilm_ctx, rpe_ctx, work_id 





RPE_HIGHRES_STEPS_1_14_26: List[ssf.RPEStepFn] = [
    ssf.step_rpe_highres_diff_enh,
    ssf.step_rpe_highres_peak_suppress_to_rpe_refined,
    ssf.step_rpe_highres_higher_res_gradient_guided_DP_to_rpe_refined2,
]

RPE_STEPS_1_14_26: List[ssf.RPEStepFn] = [
    ssf.step_rpe_init_working,
    ssf.step_rpe_hypersmoother,
    ssf.step_rpe_downsample_and_preprocess,
    ssf.step_rpe_compute_enhancement,
    ssf.step_rpe_peak_suppression,
    ssf.step_rpe_recalculate_single_seeded_and_reseed,
    ssf.step_rpe_paths_prob_edge,
    ssf.step_rpe_extract_rpe_raw_and_margin,
    ssf.step_rpe_guided_dp,
    ssf.step_rpe_tube_smoother,
    # ssf.ckpt(ssf.step_rpe_smooth_and_upsample,overwrite=False,save_by_ID=True),
    ssf.ckpt(ssf.step_rpe_upsample,overwrite=True,save_by_ID=True),
    # ssf.step_rpe_smooth_and_upsample,
] + RPE_HIGHRES_STEPS_1_14_26 + [ ssf.step_rpe_unsmooth,]

RPE_STEPS_1_14_26 = ssf.filter_pipeline(RPE_STEPS_1_14_26 )


RPE_STEPS_1_25_26: List[ssf.RPEStepFn] = [
    ssf.step_rpe_init_working,
    ssf.step_rpe_hypersmoother,
    ssf.step_rpe_downsample_and_preprocess,
    ssf.step_rpe_compute_enhancement,
    ssf.step_rpe_peak_suppression,
    ssf.step_rpe_recalculate_single_seeded_and_reseed,
    ssf.step_rpe_paths_prob_edge,
    ssf.step_rpe_extract_rpe_raw_and_margin,
    ssf.step_rpe_guided_dp,
    ssf.step_rpe_tube_smoother,
    ssf.step_rpe_upsample,
    # ssf.ckpt(ssf.step_rpe_unsmooth,overwrite=False,save_by_ID=True),
    ssf.step_rpe_unsmooth,
] + [ssf.step_rpe_highres_smooth]  + RPE_HIGHRES_STEPS_1_14_26 + [ssf.step_rpe_highres_unsmooth ]

RPE_STEPS_1_25_26 = ssf.filter_pipeline(RPE_STEPS_1_25_26 )

# print(f"RPE_STEPS is equal to {RPE_STEPS_1_25_26 }")


RPE_STEPS_2_2_26: List[ssf.RPEStepFn] = [
    ssf.step_rpe_init_working,
    ssf.step_rpe_hypersmoother,
    ssf.step_rpe_downsample_and_preprocess,
    ssf.step_rpe_compute_enhancement,
    ssf.step_rpe_peak_suppression,
    ssf.step_rpe_recalculate_single_seeded_and_reseed,
    ssf.step_rpe_paths_prob_edge,
    ssf.step_rpe_extract_rpe_raw_and_margin,
    ssf.step_rpe_guided_dp,
    ssf.step_rpe_tube_smoother,
    ssf.step_rpe_upsample,
    ssf.step_rpe_unsmooth,
    ssf.step_rpe_highres_smooth,
    ssf.step_rpe_highres_diff_enh,
    ssf.step_rpe_highres_peak_suppress_to_rpe_refined,
    # Steps don't have to be planned sequential
    # ssf.ckpt(ssf.step_rpe_highres_higher_res_gradient_guided_DP_to_rpe_refined2,overwrite=True,save_by_ID=True),
    ssf.step_rpe_highres_higher_res_gradient_guided_DP_to_rpe_refined2,
    ssf.step_rpe_highres_DP2, # Will terminate here
    # ssf.step_rpe_highres_higher_res_gradient_guided_DP_to_rpe_refined2,
] 

RPE_STEPS_2_2_26 = ssf.filter_pipeline(RPE_STEPS_2_2_26 )


# print(f"RPE_STEPS is equal to {RPE_STEPS_1_25_26 }")


# This is a failed experiment. 
RPE_STEPS_2_10_26: List[ssf.RPEStepFn] = [ # Exploring the addition of a horizontal gradient
    ssf.step_rpe_init_working,
    ssf.step_rpe_hypersmoother,
    ssf.step_rpe_downsample_and_preprocess,
    ssf.step_rpe_compute_enhancement,
    ssf.step_rpe_peak_suppression,
    ssf.step_rpe_recalculate_single_seeded_and_reseed,
    ssf.step_rpe_paths_prob_edge,
    ssf.step_rpe_extract_rpe_raw_and_margin,
    ssf.step_rpe_guided_dp,
    ssf.step_rpe_tube_smoother,
    ssf.step_rpe_upsample,
    ssf.step_rpe_unsmooth,
    ssf.step_rpe_highres_smooth,
    ssf.step_rpe_highres_diff_enh,
    ssf.step_rpe_highres_peak_suppress_to_rpe_refined,
    ssf.ckpt(ssf.step_rpe_highres_higher_res_gradient_guided_DP_to_rpe_refined2,overwrite=False,save_by_ID=True),
    # ssf.step_rpe_highres_DP2, # Will terminate here
    # ssf.step_rpe_highres_higher_res_gradient_guided_DP_to_rpe_refined2
    ssf.step_rpe_highres_grad_testing,
]

RPE_STEPS_2_10_26 = ssf.filter_pipeline(RPE_STEPS_2_10_26 )


# Testing the sticky lower line
# Seems to work!
RPE_STEPS_2_11_26: List[ssf.RPEStepFn] = [ # Exploring the addition of a horizontal gradient
    ssf.step_rpe_init_working,
    ssf.step_rpe_hypersmoother,
    ssf.step_rpe_downsample_and_preprocess,
    ssf.step_rpe_compute_enhancement,
    ssf.step_rpe_peak_suppression,
    ssf.step_rpe_recalculate_single_seeded_and_reseed,
    ssf.step_rpe_paths_prob_edge,
    ssf.step_rpe_extract_rpe_raw_and_margin,
    ssf.step_rpe_guided_dp,
    ssf.step_rpe_tube_smoother,
    ssf.step_rpe_upsample,
    ssf.step_rpe_unsmooth,
    ssf.step_rpe_highres_smooth,
    ssf.step_rpe_highres_diff_enh,
    ssf.step_rpe_highres_peak_suppress_to_rpe_refined,
    ssf.ckpt(ssf.step_rpe_highres_higher_res_gradient_guided_DP_to_rpe_refined2,overwrite=False,save_by_ID=True),
    # ssf.step_rpe_highres_DP2, # Will terminate here
    # ssf.step_rpe_highres_higher_res_gradient_guided_DP_to_rpe_refined2
    # ssf.step_rpe_highres_DP2,
    ssf.step_rpe_highres_DP_two_layer,
]

RPE_STEPS_2_11_26 = ssf.filter_pipeline(RPE_STEPS_2_11_26 )

RPE_STEPS_2_12_26: List[ssf.RPEStepFn] = [ # Exploring the addition of a horizontal gradient
    ssf.step_rpe_init_working,
    ssf.step_rpe_hypersmoother,
    ssf.step_rpe_downsample_and_preprocess,
    ssf.step_rpe_compute_enhancement,
    ssf.step_rpe_peak_suppression,
    ssf.step_rpe_recalculate_single_seeded_and_reseed,
    ssf.step_rpe_paths_prob_edge,
    ssf.step_rpe_extract_rpe_raw_and_margin,
    ssf.step_rpe_guided_dp,
    ssf.step_rpe_tube_smoother,
    ssf.step_rpe_upsample,
    ssf.step_rpe_unsmooth,
    ssf.step_rpe_highres_smooth,
    ssf.step_rpe_highres_diff_enh,
    ssf.step_rpe_highres_peak_suppress_to_rpe_refined,
    ssf.step_rpe_highres_higher_res_gradient_guided_DP_to_rpe_refined2,
    # ssf.step_rpe_highres_DP2, # Will terminate here
    # ssf.step_rpe_highres_higher_res_gradient_guided_DP_to_rpe_refined2
    # ssf.step_rpe_highres_DP2,
    # ssf.ckpt(ssf.step_rpe_highres_DP_two_layer,overwrite=False,save_by_ID=True),
    ssf.step_rpe_highres_DP_two_layer,
    ssf.step_rpe_highres_unsmooth,
    # ssf.step_rpe_endpoint_plot,
]

RPE_STEPS_2_12_26 = ssf.filter_pipeline(RPE_STEPS_2_12_26 )


RPE_STEPS_2_14_26  = RPE_STEPS_2_12_26[:-2] + [ssf.ckpt(ssf.step_rpe_highres_DP_two_layer,overwrite=True,save_by_ID=True),ssf.step_rpe_highres_unsmooth,ssf.step_rpe_endpoint_plot]
RPE_STEPS_2_14_26 = ssf.filter_pipeline(RPE_STEPS_2_14_26 )


RPE_STEPS_2_14_26_debug  = RPE_STEPS_2_12_26[:-3] + [ssf.ckpt(ssf.step_rpe_highres_higher_res_gradient_guided_DP_to_rpe_refined2,overwrite=True,save_by_ID=True),ssf.step_rpe_highres_DP2_debug]
RPE_STEPS_2_14_26_debug  = ssf.filter_pipeline(RPE_STEPS_2_14_26_debug  )



# Trialing this after it seems the seed and grab bottom line approach is actually too fragile. 
# Key logic is that all the rpe_smooth does (low-res pathway) is flatten for the highres. when it goes into choroid, it pulls the choroid up into a flat segment, and keeps it indefinitely for the highres path
# As it is never actually used to guide subsequent steps, it wouldn't be an issue if it were localized higher or lower. I suspect either grabbing top of the ctx.enh via another gradient, oor grabbing brightest path thru will
#be better and avoid grabbing RPE. The key is we don't actually need to grab the lowest line right away, and instead flattening better allows us to sweep away the choroidal texture
RPE_STEPS_2_27_26: List[ssf.RPEStepFn] = [ # Exploring the addition of a horizontal gradient
    ssf.step_rpe_init_working,
    ssf.step_rpe_hypersmoother,
    ssf.step_rpe_downsample_and_preprocess,
    ssf.step_rpe_compute_enhancement,
    ssf.step_rpe_peak_suppression,
    ssf.ckpt(ssf.step_rpe_recalculate_single_seeded_and_reseed,overwrite=True,save_by_ID=True), # getting rd of this might be a problem, as this effectively unsuppresses some over-suppressed peaks
    ssf.step_rpe_DP_on_enh_1,

    ssf.step_rpe_upsample,
    ssf.step_rpe_unsmooth,
    ssf.step_rpe_highres_smooth,
    ssf.step_rpe_highres_diff_enh,
    ssf.step_rpe_highres_peak_suppress_to_rpe_refined,
    ssf.step_rpe_highres_higher_res_gradient_guided_DP_to_rpe_refined2,
    # ssf.step_rpe_highres_DP2, # Will terminate here
    # ssf.step_rpe_highres_higher_res_gradient_guided_DP_to_rpe_refined2
    # ssf.step_rpe_highres_DP2,
    # ssf.ckpt(ssf.step_rpe_highres_DP_two_layer,overwrite=False,save_by_ID=True),
    ssf.step_rpe_highres_DP_two_layer,
    ssf.step_rpe_highres_unsmooth,
    ssf.step_rpe_endpoint_plot
    # ssf.step_rpe_paths_prob_edge, # this is the real logic we are replacing
    # ssf.step_rpe_extract_rpe_raw_and_margin, # also gtting replaced
    # ssf.step_rpe_guided_dp, #This is what gets rid of big jumps where the lowest line goes way too low to S.C. junction
    # ssf.step_rpe_tube_smoother, #this then runs in the original image space, seeking bright line
    # ssf.step_rpe_smooth_and_upsample,
    # ssf.step_rpe_unsmooth,
    # ssf.step_rpe_highres_smooth,
    # ssf.step_rpe_highres_diff_enh,
    # ssf.step_rpe_highres_peak_suppress_to_rpe_refined,
    # ssf.step_rpe_highres_higher_res_gradient_guided_DP_to_rpe_refined2,
    # # ssf.step_rpe_highres_DP2, # Will terminate here
    # # ssf.step_rpe_highres_higher_res_gradient_guided_DP_to_rpe_refined2
    # # ssf.step_rpe_highres_DP2,
    # # ssf.ckpt(ssf.step_rpe_highres_DP_two_layer,overwrite=False,save_by_ID=True),
    # ssf.step_rpe_highres_DP_two_layer,
    # ssf.step_rpe_highres_unsmooth,
    # ssf.step_rpe_endpoint_plot,
]
RPE_STEPS_2_27_26 = ssf.filter_pipeline(RPE_STEPS_2_27_26 )

ILM_STEPS_2_27_26 : List[ssf.ILMStepFn] = [
    ssf.step_ilm_downsample_and_preprocess,
    ssf.step_ilm_compute_enhancement,
    ssf.step_ilm_seed_detection,
    ssf.step_ilm_edge_impute,
    ssf.step_ilm_extract_raw,
    ssf.step_ilm_tube_smoother,
    ssf.step_ilm_upsample,
    ssf.step_ilm_endpoint_plot,
]
ILM_STEPS_2_27_26_take2 : List[ssf.ILMStepFn] = [
    ssf.step_ilm_hypersmoother,
    ssf.step_ilm_downsample_and_preprocess,
    ssf.step_ilm_compute_enhancement,
    ssf.step_ilm_peak_suppression,
    ssf.step_ilm_ax_grad_thinner,
    ssf.step_ilm_DP,
    ssf.step_ilm_DP_refiner,
    # ssf.step_ilm_DP_debug,
    ssf.step_ilm_upsample,
    ssf.step_ilm_unsmooth,
    ssf.step_ilm_endpoint_plot,
]


# We have a superiorly promising result above. Now -- add in some DBF wtih per column normalization to our initial DP on the enh
# A bigger fix might be to use the ENH_f with some gentle peak suppression. Here we test this while ignoring the highres for now to save time!
RPE_STEPS_2_28_26: List[ssf.RPEStepFn] = [ # Exploring the addition of a horizontal gradient
    ssf.step_rpe_init_working,
    ssf.step_rpe_hypersmoother,
    ssf.step_rpe_downsample_and_preprocess,
    ssf.step_rpe_compute_enhancement2,
    ssf.step_rpe_DP_on_enh_2,
    ssf.step_rpe_upsample,
    ssf.step_rpe_unsmooth,

    ssf.step_rpe_highres_smooth,
    ssf.step_rpe_highres_diff_enh,
    ssf.step_rpe_highres_peak_suppress_to_rpe_refined,
    ssf.step_rpe_highres_higher_res_gradient_guided_DP_to_rpe_refined2,
    ssf.step_rpe_highres_DP_two_layer,
    ssf.step_rpe_highres_unsmooth,
    ssf.step_rpe_endpoint_plot,
]
RPE_STEPS_2_28_26 = ssf.filter_pipeline(RPE_STEPS_2_28_26 )

ILM_STEPS_2_28 : List[ssf.ILMStepFn] = [
    ssf.step_ilm_hypersmoother,
    ssf.step_ilm_downsample_and_preprocess,
    ssf.step_ilm_compute_enhancement,
    ssf.step_ilm_peak_suppression,
    ssf.step_ilm_ax_grad_thinner,
    ssf.step_ilm_DP,
    ssf.step_ilm_DP_refiner,
    ssf.step_ilm_upsample,
    ssf.step_ilm_unsmooth,
    ssf.step_ilm_endpoint_plot,
]