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
from code_files.texture_package_prod.texture_enface_utils import retinal_thickness_map
from texture_package_prod.simulation_utils import simulate_oct_volume
from texture_package_prod.texture_io import load_layers_npz # load_landmark_dict,
from texture_package_prod.texture_plotting_utils import plot_regions_overlay, plot_feature_mosaic
from texture_package_prod.texture_regions import summarize_by_regions

from texture_package_prod.texture_extraction_utilities import (
    GLCMParams,
    TextureSweepParams,
    resample_map_to_image,
    compute_bscan_texture_volumes_to_compact_zarr,
)

# from texture_package_prod.vessel_texture_postproc_utils import estimate_vessel_mask_from_enface, postprocess_feature_dict

from code_files import file_utils
from code_files import zarr_file_utils as zfu
import time


DEFAULT_FAMILIES = (
    'firstorder',
    'heterogeneity',
    'band_energy',
    'glcm',
    'glrlm', 'glszm', 'gldm', 'ngtdm', 'lbp', 
    'gradient',
)



DEFAULT_GLCM_PARAMS = GLCMParams(
    distances=(1,),
    angles_deg=(0.0, 45.0, 90.0, 135.0),
    aggregate_distances=False,
    aggregate_angles=True,
)

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


def _parse_features_to_keep(text: str | None):
    if text is None or text.strip() == "":
        return None
    return tuple(x.strip() for x in text.split(",") if x.strip())

# def run_oct(args):
#     if args.demo_oct:
#         volume, upper, lower, _ = simulate_oct_volume(
#             z=args.demo_z, height=args.demo_h, width=args.demo_w, seed=args.seed, eye=args.demo_eye, pattern=args.demo_pattern
#         )
#     else:
#         volume = load_ss_volume(args.input, mmap=True, z_step=args.z_step)
#         upper, lower = _load_bounds(args)
#         if args.z_step > 1:
#             upper = upper[::args.z_step]
#             lower = lower[::args.z_step]


#     maps = project_bscan_texture_to_enface(
#         volume,
#         upper_bound=upper,
#         lower_bound=lower,
#         z_step=1,
#         window=args.window, #Will fail
#         step=args.step,
#         pad=args.pad,
#         families=DEFAULT_FAMILIES,
#         include_wavelet=not args.no_wavelet,
#         n_jobs=args.n_jobs,
#     )
#     out_dir = Path(args.out_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)
#     np.savez_compressed(out_dir / 'enface_maps.npz', **maps)
#     preview = np.nanmean(volume, axis=1)
#     pd.DataFrame([{'feature': k, 'mean': float(np.nanmean(v)), 'std': float(np.nanstd(v))} for k, v in maps.items()]).to_csv(out_dir / 'enface_summary.csv', index=False)
#     plot_feature_mosaic(preview, maps, meta=None, out_path=out_dir / 'enface_mosaic.png')

def run_oct_texture_pipeline(args):
    import zarr

    t1 = time.time()
    print('flattening here')
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

    volume_out_dir = Path(args.out_dir) / Path(args.input).stem
    volume_out_dir.mkdir(parents=True, exist_ok=True)

    root = zarr.open_group(str(art["image_zarr"]), mode="r")
    volume = root["data"]

    sweep_params = _build_texture_sweep_params(args)
    all_cases = [tp.concrete() for tp in sweep_params.iter_cases()]
    manifest = []

    features_to_keep = _parse_features_to_keep(args.features_to_keep)

    for texture_params in all_cases:
        run_tag = texture_params.tag()
        run_dir = volume_out_dir / run_tag
        run_dir.mkdir(parents=True, exist_ok=True)

        t1 = time.time()
        print(f"now computing compact textures for {run_tag}")

        compact_zarr_path = run_dir / 'texture_bscan_maps_compact.zarr'

        compute_bscan_texture_volumes_to_compact_zarr(
            volume=volume,
            upper_bound=upper,
            lower_bound=lower,
            out_zarr_path=compact_zarr_path,
            z_step=1,
            step=args.step,
            pad=args.pad,
            families=DEFAULT_FAMILIES,
            include_wavelet=not args.no_wavelet,
            texture_params=texture_params,
            features_to_keep=features_to_keep,
            n_jobs=args.n_jobs,
            single_bscan_n_jobs=args.single_bscan_n_jobs,
            overwrite=args.overwrite,
            glcm_params=DEFAULT_GLCM_PARAMS,
        )

        print(f"we have completed the compact zarr textures for {run_tag}: {time.time()-t1} time")

        with open(run_dir / 'compact_zarr_path.txt', 'w') as f:
            f.write(str(compact_zarr_path) + '\n')

        with open(run_dir / 'texture_params.json', 'w') as f:
            json.dump(texture_params.as_attrs(), f, indent=2)

        manifest.append({
            'tag': run_tag,
            'texture_output_format': 'compact_zarr',
            'compact_zarr_path': str(compact_zarr_path),
            'flat_image_zarr': str(art["image_zarr"]),
            'flat_layers_npz': str(art["flat_layers_npz"]),
            'input_volume': str(args.input),
            'texture_params': texture_params.as_attrs(),
            'rpe_offset': int(args.rpe_offset),
            'slab_thickness': int(args.slab_thickness),
            'pad': int(args.pad),
            'step': int(args.step),
            'features_to_keep': None if features_to_keep is None else list(features_to_keep),

            'glcm_params': {
                'distances': [int(x) for x in DEFAULT_GLCM_PARAMS.distances],
                'angles_deg': [float(x) for x in DEFAULT_GLCM_PARAMS.angles_deg],
                'aggregate_distances': bool(DEFAULT_GLCM_PARAMS.aggregate_distances),
                'aggregate_angles': bool(DEFAULT_GLCM_PARAMS.aggregate_angles),
            },
        })

    with open(volume_out_dir / 'texture_runs_manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)

    with open(volume_out_dir / 'texture_paths.txt', 'w') as f:
        for row in manifest:
            f.write(f"{row['tag']}\t{row['compact_zarr_path']}\n")

    if len(manifest) == 1:
        with open(volume_out_dir / 'compact_zarr_path.txt', 'w') as f:
            f.write(str(manifest[0]['compact_zarr_path']) + '\n')


def make_argparser():
    p = argparse.ArgumentParser(description='Readable texture-analysis entry point.')
    p.add_argument('--input', type=str, default='')
    p.add_argument('--out-dir', type=str, required=True)
    # p.add_argument('--window', type=int, default=31)
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


    p.add_argument('--features_to_keep', type=str, default=None)


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
    run_oct_texture_pipeline(args)


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