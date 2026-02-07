
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

import file_utils as fu
import code_files.segmentation_code.segmentation_pipelines as sp


# ---------------------------- lite extraction ----------------------------

def extract_lite(ilm_ctx, rpe_ctx):
    """Return lightweight dict: 1D lines + a few small attrs (no big images)."""
    hp = rpe_ctx.hypersmoother_params

    hr = getattr(rpe_ctx, "highres_ctx", None)

    d = {
        # ---- final-ish lines you want stackable ----
        "hypersmoother_path": hp.hypersmoother_path,
        "rpe_raw": rpe_ctx.rpe_raw,
        "rpe_smooth": rpe_ctx.rpe_smooth,
        "ilm_smooth": ilm_ctx.ilm_smooth,

        # ---- optional refined lines ----
        "rpe_refined1": hr.rpe_refined,
        "rpe_refined2": hr.rpe_refined2,

        # ---- light params ----
        # keep shift if you want resume/unsmooth later
        "hypersmoother_target_y": hp.hypersmoother_target_y,
        "hypersmoother_shift_y_full": hp.hypersmoother_shift_y_full,
        "hypersmoother_target_y":hp.hypersmoother_target_y,

        "highres_smoother_shift_y_full":hp.highres_smoother_shift_y_full,
        "highres_smoother_target_y":hp.highres_smoother_target_y,

    }
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

def load_vol_and_onh(vol_fp):
    vol_fp = Path(vol_fp)
    vol = fu.load_ss_volume2(vol_fp, mmap=True)

    onh_path = fu.image_to_annotation_path(vol_fp)
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

def process_volume_lite(vol_fp, *, z_step=1, max_workers=8, rpe_steps=None, out_dir=None):
    vol_fp = Path(vol_fp)
    vol_id = vol_fp.with_suffix("").name

    vol, onh = load_vol_and_onh(vol_fp)

    z_idx = np.arange(0, vol.shape[0], int(z_step))
    work = build_work(vol, onh, z_idx, vol_id)

    vol_out = Path(out_dir) / vol_id
    vol_out.mkdir(parents=True, exist_ok=True)

    fn = sp.process_bscan_1_3_26  # must return (idx, ilm_ctx, rpe_ctx) when production_mode=False

    if max_workers <= 1:
        results = [fn(t, False, rpe_steps) for t in work]
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(fn, t, False, rpe_steps) for t in work]
            results = [f.result() for f in futs]

    results.sort(key=lambda x: x[0])

    slice_dicts = []
    for k, ilm_ctx, rpe_ctx, work_id in results:
        z = int(z_idx[k])
        d = extract_lite(ilm_ctx, rpe_ctx)

        _save_npz(vol_out / f"z{z:04d}.npz", d)
        slice_dicts.append((z, d))

    STACK_KEYS = [
        "hypersmoother_path",
        "rpe_raw",
        "rpe_smooth",
        "ilm_smooth",
        "rpe_refined1",
        "rpe_refined2",
        # add "rpe_smooth2" if you create it later
    ]
    stacked = collate_stackable(slice_dicts, STACK_KEYS)
    np.savez_compressed(vol_out / f"{vol_id}_stacked.npz", **stacked)

    return vol_out


def batch_process_dir_lite(volumes_root, *, pattern="*.npy",
                           z_step=1, max_workers=8, rpe_steps=None,
                           outputs_root=None):
    volumes_root = Path(volumes_root)

    if outputs_root is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        outputs_root = Path("segmentation_outputs") / run_id
    outputs_root = Path(outputs_root)
    outputs_root.mkdir(parents=True, exist_ok=True)

    for vol_path in sorted(volumes_root.glob(pattern)):
        print("Processing", vol_path.name, flush=True)
        out = process_volume_lite(
            vol_path, 
            z_step=z_step, max_workers=max_workers,
            rpe_steps=rpe_steps, out_dir=outputs_root
        )
        print("saved ->", out)

    print("DONE. outputs_root =", outputs_root)
    return outputs_root


# ---------------------------- CLI ----------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--volumes_root", type=str)
    parser.add_argument("--pattern", type=str, default=".img")
    parser.add_argument("--z_step", type=int, default=1)
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--outputs_root", type=str)
    args = parser.parse_args()

    STEPS = sp.RPE_STEPS_1_25_26  # swap to your desired list

    batch_process_dir_lite(
        args.volumes_root,
        pattern=args.pattern,
        z_step=args.z_step,
        max_workers=args.max_workers,
        rpe_steps=STEPS,
        outputs_root=args.outputs_root,
    )

















