
#!/usr/bin/env python3
import sys, math
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import dask.array as da

# --- project path ---
sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[2]))

from code_files.segmentation_code.segmentation_step_functions import RPEContext
import file_utils as fu
import code_files.segmentation_code.segmentation_pipelines as sp


# ---------------------------- lite extraction ----------------------------

def extract_lite(ilm_ctx, rpe_ctx):
    """Return lightweight dict: 1D lines + a few small attrs (no big images)."""
    hp = rpe_ctx.hypersmoother_params

    hr = getattr(rpe_ctx, "highres_ctx", None)

    d = dict(
        ## ---- final-ish lines you want stackable ----
        hypersmoother_path = hp.hypersmoother_path,
        rpe_raw = rpe_ctx.rpe_raw,
        rpe_smooth = rpe_ctx.rpe_smooth,
        ilm_smooth = ilm_ctx.ilm_smooth,
        #  ----optional refined lines ----
        rpe_refined1 = hr.rpe_refined,
        rpe_refined2 = hr.rpe_refined2,
        rpe_smooth2 = hr.rpe_smooth2,
        #  ----light params ----
        #  keepshift if you want resume/unsmooth later
        hypersmoother_target_y =  hp.hypersmoother_target_y,
        hypersmoother_shift_y_full =  hp.hypersmoother_shift_y_full,
        # hypersmoother_target_y = hp.hypersmoother_target_y,
        highres_smoother_shift_y_full = hp.highres_smoother_shift_y_full,
        highres_smoother_target_y = hp.highres_smoother_target_y,

    )

    tl_names = dict(
        original_method='two_layer_dp_ctx',
        choroidal_method = 'two_layer_dp_ctx_choroidal',
        EZ_method = 'two_layer_dp_ctx_EZ')
    for n,attr in tl_names.items():
        tl = getattr(rpe_ctx, attr, None)

        d[f'{n}_y1_rescaled'] = tl.y1_rescaled
        d[f'{n}_y2_rescaled'] = tl.y2_rescaled
        d[f'{n}_y1_vertical_shifted'] = tl.y1_vertical_shifted
        d[f'{n}_y2_vertical_shifted'] = tl.y2_vertical_shifted

    return d


def _save_npz(path, d):
    # np.savez_compressed doesn't like None -> drop them
    np.savez_compressed(path, **{k: v for k, v in d.items() if v is not None})


def collate_stackable(slice_dicts, keys):
    """
    slice_dicts: list of (z, dict) in increasing z
    keys: which keys to try stacking into (n_slices, W)
    """
    zs = [z for z, _ in slice_dicts]
    out = {"z": np.asarray(zs, dtype=np.int32)}

    # pick reference width Wref from first available 1D array among keys
    Wref = None
    for _, d in slice_dicts:
        for k in keys:
            a = d.get(k, None)
            if a is not None:
                a = np.asarray(a)
                if a.ndim == 1:
                    Wref = int(a.shape[0])
                    break
        if Wref is not None:
            break

    for k in keys:
        arrs = []
        for _, d in slice_dicts:
            a = d.get(k, None)
            if a is None:
                continue
            a = np.asarray(a)
            if a.ndim == 1 and a.shape[0] == Wref:
                arrs.append(a)
        if arrs:
            out[k] = np.stack(arrs, axis=0)

    return out


# ---------------------------- IO helpers ----------------------------

def load_vol_and_onh(vol_fp,annotation_root):
    vol_fp = Path(vol_fp)
    vol = fu.load_ss_volume2(vol_fp, mmap=True)

    onh_path = fu.image_to_annotation_path(vol_fp,annotation_root)
    onh = da.from_zarr(onh_path)
    # if onh.shape != vol.shape:
    #     onh = subsample_volume(onh, vol.shape[0])
    return vol, onh


def build_work(vol, onh, z_idx, vol_id):
    work = []
    for k, z in enumerate(z_idx):
        bscan = vol[z, :, :]
        onh_z = onh[z, :, :][...]
        work.append((k, bscan, onh_z, f"{vol_id}_z:{int(z)}"))
    return work


# ---------------------------- main processing ----------------------------

def process_volume_lite(vol_fp, *, z_step=1, max_workers=8, rpe_steps=None,ilm_steps=None, out_dir=None,annotation_root=None):
    vol_fp = Path(vol_fp)
    vol_id = vol_fp.with_suffix("").name

    # vol, onh = load_vol_and_onh(vol_fp,annotation_root)
    vol, onh = fu.load_vol_and_annotation(vol_fp,annotation_root)

    z_idx = np.arange(0, vol.shape[0], int(z_step))
    work = build_work(vol, onh, z_idx, vol_id)

    vol_out = Path(out_dir) / vol_id
    vol_out.mkdir(parents=True, exist_ok=True)

    fn = sp.process_bscan_1_3_26  # must return (idx, ilm_ctx, rpe_ctx) when production_mode=False

    if max_workers <= 1:
        results = [fn(t, False, rpe_steps,ilm_steps) for t in work]
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(fn, t, False, rpe_steps,ilm_steps) for t in work]
            results = [f.result() for f in futs]

    results.sort(key=lambda x: x[0])

    slice_dicts = []
    for k, ilm_ctx, rpe_ctx, work_id in results:
        z = int(z_idx[k])
        d = extract_lite(ilm_ctx, rpe_ctx)

        _save_npz(vol_out / f"z{z:04d}.npz", d)
        slice_dicts.append((z, d))

    # STACK_KEYS = [
    #     "hypersmoother_path",
    #     "rpe_raw",
    #     "rpe_smooth",
    #     "ilm_smooth",
    #     "rpe_refined1",
    #     "rpe_refined2",
    #     "rpe_smooth2",
    #     'y1_rescaled',
    #     'y2_rescaled',
    #     # add "rpe_smooth2" if you create it later
    # ]
    # Now on 3/19/26 we have the three-fold path. Will keep the rescaled along w/ the shifted as well.
    STACK_KEYS = [
        "hypersmoother_path",
        "rpe_raw",
        "rpe_smooth",
        "ilm_raw",
        "ilm_smooth",

        'original_method_y1_rescaled',
        'original_method_y2_rescaled',
        'original_method_y1_vertical_shifted',
        'original_method_y2_vertical_shifted',
        'choroidal_method_y1_rescaled',
        'choroidal_method_y2_rescaled',
        'choroidal_method_y1_vertical_shifted',
        'choroidal_method_y2_vertical_shifted',
        'EZ_method_y1_rescaled',
        'EZ_method_y2_rescaled',
        'EZ_method_y1_vertical_shifted',
        'EZ_method_y2_vertical_shifted',
    ]


    stacked = collate_stackable(slice_dicts, STACK_KEYS)
    np.savez_compressed(vol_out / f"{vol_id}_stacked.npz", **stacked)

    return vol_out


def batch_process_dir_lite(ALL_VOL_PATHS, rpe_steps,ilm_steps,
                           outputs_root,
                           annotation_root,
                           *, 
                           z_step=1, max_workers=8, 
                           ):
    # volumes_root = Path(volumes_root)
    if outputs_root is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        outputs_root = Path("segmentation_outputs") / run_id
    outputs_root = Path(outputs_root)
    outputs_root.mkdir(parents=True, exist_ok=True)

    # ALL_VOL_PATHS = fu.get_all_vol_paths(volumes_root,pattern,)
    for vol_path in sorted(ALL_VOL_PATHS):
        print("Processing", vol_path.name, flush=True)
        out = process_volume_lite(
            vol_path, 
            z_step=z_step, max_workers=max_workers,
            rpe_steps=rpe_steps, ilm_steps=ilm_steps,out_dir=outputs_root,
            annotation_root=annotation_root
        )
        print("saved ->", out)

    print("DONE. outputs_root =", outputs_root)
    return outputs_root


# ---------------------------- CLI ----------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--volumes_root", type=str)
    parser.add_argument("--pattern", type=str, default=None)
    parser.add_argument("--z_step", type=int, default=1)
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--outputs_root", type=str)
    parser.add_argument("--annotation_root", type=str)
    parser.add_argument("--cube_numbers", type=str, default=None)
    parser.add_argument("--input_file", type=str, default=None,help="Takes a single filename (not full path), and segments just that one")
    args = parser.parse_args()

    # STEPS = sp.RPE_STEPS_1_25_26  # swap to your desired list
    # STEPS = sp.RPE_STEPS_2_12_26  # swap to your desired list
    ILM_STEPS = sp.ILM_STEPS_2_28  # swap to your desired list
    # RPE_STEPS = sp.RPE_STEPS_2_28_26  # swap to your desired list
    RPE_STEPS = sp.RPE_STEPS_unified_3_19_26  # swap to your desired list

    ALL_VOL_PATHS = fu.get_all_vol_paths(args.volumes_root,args.pattern,args.cube_numbers)
    print(f"going to be processing {ALL_VOL_PATHS}")
    batch_process_dir_lite(
        ALL_VOL_PATHS,
        rpe_steps=RPE_STEPS,
        ilm_steps=ILM_STEPS,
        z_step=args.z_step,
        max_workers=args.max_workers,
        outputs_root=args.outputs_root,
        annotation_root=args.annotation_root,
    )



"""
--vol_dir --z_stride 25 --labels_dir_suffix _layers_2026_01_31 --cube_numbers 2,3,14

example run

python code_files/setup_data/02_segment_ILM_RPE.py --volumes_root /Volumes/T9/iowa_research/Han_AIR_Dec_2025/data_volumes/data_all_volumes2 --pattern "*.img" --z_step 750 --max_workers 8 --outputs_root /Volumes/T9/iowa_research/Han_AIR_Dec_2025/local_layers_dir/test_layers_3_19_26 --annotation_root /Users/matthewhunt/Research/Iowa_Research/Han_AIR/annotations_dir/full_annotations_2_19_26 --cube_numbers 1


"""

# python code_files/setup_data/02_segment_ILM_RPE.py \
    # --volumes_root "/Volumes/msh_uiowa/Research Data/Han_AIR_Dec_2025/data_volumes/data_all_volumes/" \
    # --pattern "*.img" \
    # --z_step 500 \
    # --max_workers 8 \
    # --outputs_root /Users/matthewhunt/Research/Iowa_Research/Han_AIR/data_volumes/data_all_volumes_layers_2026_02_12















