import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # adds Han_AIR/ to path
sys.path.append(str(Path(__file__).resolve().parents[2]))  # adds Han_AIR/ to path
import file_utils as fu
import numpy as np
from datetime import date
import code_files.segmentation_code.segmentation_pipelines as sp
from code_files.segmentation_code.segmentation_step_functions import RPEContext
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
from dataclasses import asdict
import dask.array as da
import math

def subsample_volume(volume,smaller_zdim):
    """used to correct the spacing of slices in ONH info"""
    step_size = int(math.ceil(volume.shape[0]/smaller_zdim))
    volume_out = volume[np.arange(0,volume.shape[0],step_size)]
    return volume_out

def process_volume(vol_fp,annotation_root):
    """input the volume fp bc now we have to load the onh annotations as well
    process a single volume and pull the ilm and rpe, which will be saved to surfaces"""
    layers_dict = {}

    vol = fu.load_ss_volume2(vol_fp,mmap=True) # should be fast and memory light?
    annotation_root = Path(annotation_root)
    annotation_path = fu.image_to_annotation_path(vol_fp,annotation_root)
    ONH_info = da.from_zarr(annotation_path) # should be fast and memory light?)
    # if step_size == 1:
    if ONH_info.shape != vol.shape:
        print(f"our ONH_info shape = {ONH_info.shape} and our volume shape = {vol.shape}. Will be subsampling")
        ONH_info = subsample_volume(ONH_info,vol.shape[0])

    with ProcessPoolExecutor(max_workers=12) as exe:
        futures = [
            # exe.submit(lsf.worker_process_bscan_layers, (idx, vol[idx, :, :],ONH_info[idx,:,:]), 1.5)
            exe.submit(sp.process_bscan_12_6_25, (idx, vol[idx, :, :],ONH_info[idx,:,:]))
            for idx in range(vol.shape[0])
            # for idx in range(14)
        ]
        # collect results and sort by bscan_idx
        results = [fut.result() for fut in futures]
        # for bscan_idx, dbg_ilm,dbg_rpe in sorted(results, key=lambda x: x[0]):
        for bscan_idx, out_dict in sorted(results, key=lambda x: x[0]):
            layers_dict[f'bscan{bscan_idx}'] = out_dict

    stacked_layers = collate_layers(layers_dict)

    return stacked_layers



def collate_layers(layers_dict):
    """
    layers_dict: { 'bscan0': RpeDebug, 'bscan1': RpeDebug, ... }
    
    Returns a dict mapping each RpeDebug field name to a stacked array
    of shape (n_slices, width).
    """
    # 1) Prepare empty lists for each field
    field_lists = defaultdict(list)  

    # 2) Iterate in sorted b-scan order to keep slice axis consistent
    for key in sorted(layers_dict.keys(), key=lambda k: int(k.replace('bscan',''))):
        # dbg: RpeDebug = layers_dict[key]
        out_dict = layers_dict[key]
        
        # asdict gives you a normal dict of { field_name: value_array }
        # for field_name, arr in asdict(dbg).items():
        for field_name, arr in out_dict.items():
            # skip any that are None (if you made fields optional)
            if arr is None:  
                continue
            field_lists[field_name].append(arr)

    # 3) Stack each list into one 2D array (slices × width)
    stacked = {}
    for field_name, list_of_arr in field_lists.items():
        # ensure every arr in list_of_arr has the same shape
        stacked[field_name] = np.stack(list_of_arr, axis=0)

    return stacked

def batch_process_dir(
    dir_path: str,
    file_ext: str = '.npy',
    annotation_root = None,
    output_dir_suffix: str = f"_layers_{date.today().strftime('%Y-%m-%d')}",
):
    """
    Walk `dir_path`, process every file ending in `file_ext` as a volume,
    and save a `{stem}_rpe.npy` array of shape (N_slices, W).
    """
    dir_path = Path(dir_path)

    for vol_path in sorted(dir_path.glob(f'*{file_ext}')):
        print(f"Processing {vol_path.name}…", end=' ', flush=True)
        stacked_layers = process_volume(vol_path,annotation_root)
        # Save name and process
        vol_dir = vol_path.parent
        vol_name = vol_path.with_suffix('').name
        processed_dir = vol_dir.with_name(vol_dir.name.strip("_mini") + output_dir_suffix)
        processed_dir.mkdir(parents=True,exist_ok=True)
        out_path = processed_dir / f"{vol_name}_layers"
        np.savez_compressed(out_path, **stacked_layers)

        print("done →", out_path.name)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="Your script description here."
    )

    parser.add_argument(
        "--annotation_root",
        type=str,
        default = "/Users/matthewhunt/Research/Iowa_Research/Han_AIR/testing_annotations",
        # required=True,
        help="Path to the input volume or data directory."
    )

    parser.add_argument(
        "--volumes_root",
        type=str,
        default='/Users/matthewhunt/Research/Iowa_Research/test_han_air_repo/SSOCT_texture_analysis/data/data_volumes_mini/',
        help="Directory where outputs will be saved."
    )
    args = parser.parse_args()

    # batch_process_dir(dir_path='/Users/matthewhunt/Research/Iowa_Research/Han_AIR/data_volumes/data_all_volumes/',file_ext='.img')
    batch_process_dir(dir_path=args.volumes_root,file_ext='.npy',annotation_root=args.annotation_root)