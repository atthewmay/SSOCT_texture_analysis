from __future__ import annotations
#REVIEWED

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent))
# from texture_package_prod.fundus_preprocessing_utils import preprocess_fundus_for_texture
from texture_package_prod.simulation_utils import simulate_oct_volume
from texture_package_prod.texture_io import load_image,  load_layers_npz, load_ss_volume # load_landmark_dict,
from texture_package_prod.texture_plotting_utils import plot_regions_overlay, plot_feature_mosaic
from texture_package_prod.texture_regions import make_etdrs_grid_plus_rings, summarize_by_regions
from texture_package_prod.texture_extraction_utilities import (
    TextureSweepParams,
    compute_dense_texture_maps,
    project_bscan_texture_to_enface,
    resample_map_to_image,
    compute_bscan_texture_volumes_to_zarr,
    retinal_thickness_map,
    # save_enface_feature_maps_to_zarr,
)

# from texture_package_prod.vessel_texture_postproc_utils import estimate_vessel_mask_from_enface, postprocess_feature_dict

from code_files import file_utils
from code_files import zarr_file_utils as zfu
import time

DEFAULT_FAMILIES = ('firstorder', 'glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm', 'lbp', 'gradient')

def _parse_csv_arg(text, cast):
    if text is None:
        return None
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if not parts:
        return None
    return tuple(cast(p) for p in parts)


def _build_texture_sweep_params(args) -> TextureSweepParams:
    windows = _parse_csv_arg(args.windows, int) or (int(args.window),)
    levels = _parse_csv_arg(args.levels_list, int) or (int(args.levels),)
    gaussian_sigmas = _parse_csv_arg(args.gaussian_sigmas, float) or (float(args.gaussian_sigma),)
    downsample_factors = _parse_csv_arg(args.downsample_factors, int) or (int(args.downsample_factor),)

    return TextureSweepParams(
        window=windows,
        levels=levels,
        gaussian_sigma=gaussian_sigmas,
        downsample_factor=downsample_factors,
    )


def _save_dense_outputs(out_dir: Path, base_image: np.ndarray, maps: dict[str, np.ndarray], meta, masks=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / 'dense_maps.npz', **maps)
    plot_feature_mosaic(base_image, maps, meta=meta, out_path=out_dir / 'feature_mosaic.png')
    if masks is not None:
        plot_regions_overlay(base_image, masks, out_path=out_dir / 'regions_overlay.png')
        rows = []
        for name, arr in maps.items():
            arr_full = arr if meta is None else resample_map_to_image(arr, meta)
            rows.append({'feature': name, **summarize_by_regions(arr_full, masks)})
        pd.DataFrame(rows).to_csv(out_dir / 'regional_summary.csv', index=False)

def _load_bounds(args):
    if args.demo_oct:
        _, upper, lower, _ = simulate_oct_volume(
            z=args.demo_z, height=args.demo_h, width=args.demo_w, seed=args.seed, eye=args.demo_eye, pattern=args.demo_pattern
        )
        return upper, lower
    if args.layers_root is None:
        raise ValueError('For OCT mode, supply --layers-npz or use --demo-oct')
    layers = load_layers_npz(args.input, args.layers_root)
    algo_key = file_utils.get_algorithm_key_from_filepath(args.input)
    lower = layers[algo_key]
    upper = layers['ilm_smooth']
    # if args.upper_key not in layers or args.lower_key not in layers:
    #     raise ValueError(f'Could not find {args.upper_key}/{args.lower_key} in {args.layers_root}')
    # return layers[args.upper_key], layers[args.lower_key]
    return upper,lower

def compute_extra_enface_feature_maps(args, layers):
    """
    Non-texture features that still belong in the same downstream feature stack.
    """
    rpe_key = file_utils.get_algorithm_key_from_filepath(args.input)

    ilm = layers['ilm_smooth']
    rpe = layers[rpe_key]

    return {
        'geometry__retinal_thickness': retinal_thickness_map(ilm, rpe),
    }


def run_oct(args):
    if args.demo_oct:
        volume, upper, lower, _ = simulate_oct_volume(
            z=args.demo_z, height=args.demo_h, width=args.demo_w, seed=args.seed, eye=args.demo_eye, pattern=args.demo_pattern
        )
    else:
        volume = load_ss_volume(args.input, mmap=True, z_step=args.z_step)
        upper, lower = _load_bounds(args)
        if args.z_step > 1:
            upper = upper[::args.z_step]
            lower = lower[::args.z_step]


    maps = project_bscan_texture_to_enface(
        volume,
        upper_bound=upper,
        lower_bound=lower,
        z_step=1,
        window=args.window,
        step=args.step,
        pad=args.pad,
        families=DEFAULT_FAMILIES,
        include_wavelet=not args.no_wavelet,
        n_jobs=args.n_jobs,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / 'enface_maps.npz', **maps)
    preview = np.nanmean(volume, axis=1)
    pd.DataFrame([{'feature': k, 'mean': float(np.nanmean(v)), 'std': float(np.nanstd(v))} for k, v in maps.items()]).to_csv(out_dir / 'enface_summary.csv', index=False)
    plot_feature_mosaic(preview, maps, meta=None, out_path=out_dir / 'enface_mosaic.png')


# def run_oct_to_zarr(args):
#     import zarr
#     # volume = load_ss_volume(args.input, mmap=True, z_step=args.z_step)
#     # upper, lower = _load_bounds(args)

#     t1 = time.time()
#     print(f'flattening here')
#     art = zfu.ensure_flattened_artifacts(
#         vol_path=Path(args.input),
#         flatten_with=file_utils.get_algorithm_key_from_filepath(args.input),
#         layers_root=args.layers_root,
#         z_stride=args.z_step,
#         overwrite=args.overwrite,
#         make_image_zarr=True,
#         make_label_zarr=True,
#         save_flat_layers_npz=True,
#     )
#     print(f"we have completed the flattening: {time.time()-t1} time")

#     layers = np.load(art['flat_layers_npz'])
#     algo_key = file_utils.get_algorithm_key_from_filepath(args.input)
#     lower = layers[algo_key] + args.rpe_offset
#     # upper = layers['ilm_smooth']
#     upper = lower-args.slab_thickness

#     out_dir = Path(args.out_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)
#     out_dir = out_dir/Path(args.input).stem

#     zarr_path = out_dir / 'texture_bscan_maps.zarr'

#     root = zarr.open_group(str(art["image_zarr"]), mode="r")
#     volume = root["data"]
#     t1 = time.time()
#     print("now computing the textures")
#     compute_bscan_texture_volumes_to_zarr(
#         volume=volume,
#         upper_bound=upper,
#         lower_bound=lower,
#         out_zarr_path=zarr_path,
#         z_step=1,
#         window=args.window,
#         step=args.step,
#         pad=args.pad,
#         families=DEFAULT_FAMILIES,
#         include_wavelet=not args.no_wavelet,
#         # features_to_keep=('raw__mean', 'raw__std', 'raw__glcm_contrast'),
#         features_to_keep=None,
#         n_jobs=args.n_jobs,
#         single_bscan_n_jobs=args.single_bscan_n_jobs,
#     )
#     print(f"we have completed the textures: {time.time()-t1} time")

#     with open(out_dir / 'zarr_path.txt', 'w') as f:
#         f.write(str(zarr_path) + '\n')

def run_oct_to_zarr(args):
    import zarr

    t1 = time.time()
    print(f'flattening here')
    art = zfu.ensure_flattened_artifacts(
        vol_path=Path(args.input),
        flatten_with=file_utils.get_algorithm_key_from_filepath(args.input),
        layers_root=args.layers_root,
        z_stride=args.z_step,
        overwrite=args.overwrite,
        make_image_zarr=True,
        make_label_zarr=True,
        save_flat_layers_npz=True,
    )
    print(f"we have completed the flattening: {time.time()-t1} time")

    layers = np.load(art['flat_layers_npz'])
    algo_key = file_utils.get_algorithm_key_from_filepath(args.input)
    lower = layers[algo_key] + args.rpe_offset
    upper = lower - args.slab_thickness

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    volume_out_dir = out_dir / Path(args.input).stem
    volume_out_dir.mkdir(parents=True, exist_ok=True)

    root = zarr.open_group(str(art["image_zarr"]), mode="r")
    volume = root["data"]

    sweep_params = _build_texture_sweep_params(args)
    manifest = []

    for texture_params in sweep_params.iter_cases():
        texture_params = texture_params.concrete()
        run_tag = texture_params.tag()
        run_dir = volume_out_dir / run_tag
        run_dir.mkdir(parents=True, exist_ok=True)

        zarr_path = run_dir / 'texture_bscan_maps.zarr'

        t1 = time.time()
        print(f"now computing textures for {run_tag}")

        compute_bscan_texture_volumes_to_zarr(
            volume=volume,
            upper_bound=upper,
            lower_bound=lower,
            out_zarr_path=zarr_path,
            z_step=1,
            step=args.step,
            pad=args.pad,
            families=DEFAULT_FAMILIES,
            include_wavelet=not args.no_wavelet,
            texture_params=texture_params,
            features_to_keep=None,
            n_jobs=args.n_jobs,
            single_bscan_n_jobs=args.single_bscan_n_jobs,
            overwrite=args.overwrite,
        )

        print(f"we have completed the textures for {run_tag}: {time.time()-t1} time")

        with open(run_dir / 'texture_params.json', 'w') as f:
            json.dump(texture_params.as_attrs(), f, indent=2)

        # manifest.append({
        #     'tag': run_tag,
        #     'zarr_path': str(zarr_path),
        #     **texture_params.as_attrs(),
        # })

        manifest.append({
                'tag': run_tag,
                'zarr_path': str(zarr_path),
                'flat_image_zarr': str(art["image_zarr"]),
                'flat_layers_npz': str(art["flat_layers_npz"]),
                'input_volume': str(args.input),
                'texture_params': texture_params.as_attrs(),
                'rpe_offset': int(args.rpe_offset),
                'slab_thickness': int(args.slab_thickness),
                'pad': int(args.pad),
                'step': int(args.step),
            })

    with open(volume_out_dir / 'texture_runs_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)

    with open(volume_out_dir / 'zarr_paths.txt', 'w') as f:
        for row in manifest:
            f.write(f"{row['tag']}\t{row['zarr_path']}\n")

    if len(manifest) == 1:
        with open(volume_out_dir / 'zarr_path.txt', 'w') as f:
            f.write(str(manifest[0]['zarr_path']) + '\n')



def make_argparser():
    p = argparse.ArgumentParser(description='Readable texture-analysis entry point.')
    p.add_argument('--input', type=str, default='')
    p.add_argument('--out-dir', type=str, required=True)
    p.add_argument('--window', type=int, default=31)
    p.add_argument('--step', type=int, default=6)
    p.add_argument('--n-jobs', type=int, default=1)
    p.add_argument('--single_bscan_n_jobs', type=int, default=1)
    p.add_argument('--no-wavelet', action='store_true')
    p.add_argument('--seed', type=int, default=0)


    p.add_argument('--levels', type=int, default=32)
    p.add_argument('--gaussian-sigma', type=float, default=0.0)
    p.add_argument('--downsample-factor', type=int, default=1)

    p.add_argument('--windows', type=str, default=None)
    p.add_argument('--levels-list', type=str, default=None)
    p.add_argument('--gaussian-sigmas', type=str, default=None)
    p.add_argument('--downsample-factors', type=str, default=None)


    p.add_argument('--landmarks', type=str, default=None)
    p.add_argument('--postprocess-vessels', action='store_true')
    p.add_argument('--post-radius', type=int, default=5)

    p.add_argument('--overwrite', action='store_true')
    

    p.add_argument('--rpe_offset', type=int, default=10)
    p.add_argument('--slab_thickness', type=int, default=50)

    p.add_argument('--layers-root', type=str, default=None)
    p.add_argument('--upper-key', type=str, default='ilm_smooth')
    p.add_argument('--lower-key', type=str, default='rpe_smooth')
    p.add_argument('--pad', type=int, default=10)
    p.add_argument('--z-step', type=int, default=1)
    p.add_argument('--demo-oct', action='store_true')
    p.add_argument('--demo-z', type=int, default=12)
    p.add_argument('--demo-h', type=int, default=128)
    p.add_argument('--demo-w', type=int, default=160)
    p.add_argument('--demo-eye', type=str, default='R')
    p.add_argument('--demo-pattern', type=str, default='focal', choices=['focal', 'banded'])
    return p


def main():
    args = make_argparser().parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out_dir) / 'run_args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    run_oct_to_zarr(args)


if __name__ == '__main__':
    main()


# """
# python -m code_files.texture_package_prod.run_texture_pipeline \
#   --mode oct \
#   --input /path/to/vol.img \
#   --out-dir /path/to/out \
#   --layers-root /path/to/layers \
#   --windows 31,41 \
#   --levels-list 16,32 \
#   --gaussian-sigmas 0,1 \
#   --downsample-factors 1,2,4
# """