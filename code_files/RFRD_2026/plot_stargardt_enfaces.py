from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr


from code_files import file_utils as fu
from code_files import zarr_file_utils as zfu
from code_files.texture_package_prod.texture_enface_utils import (
    compute_extra_enface_maps,
    make_enface_dict_isotropic_x,
)


VOL_ROOT = Path("/Volumes/T9/iowa_research/Han_stargardt_2026/data_volumes")
LAYERS_ROOT = Path("/Volumes/T9/iowa_research/Han_stargardt_2026/layers_5_12")
OUT_ROOT = Path("/Volumes/T9/iowa_research/Han_stargardt_2026/enface_maps/")

SLAB_OFFSETS = ((-30,-10),(-10,10),(0, 10),(10, 20),(20,40),(0,50),(0,100))
GLOB = "*.img"
OVERWRITE_FLAT = False

stargardt_layer_to_use = 'original_method_y2_vertical_shifted'


def save_one(vol_path: Path):
    if "stargardt" in str(vol_path):
        print("using hard coded original layer algo")
        reference_key = stargardt_layer_to_use
    else:
        print("using ID2algo map")
        reference_key = fu.get_algorithm_key_from_filepath(vol_path)

    art = zfu.ensure_flattened_artifacts(
        vol_path=vol_path,
        flatten_with=reference_key,
        layers_root=LAYERS_ROOT,
        z_stride=1,
        overwrite=OVERWRITE_FLAT,
        make_image_zarr=True,
        make_label_zarr=False,
        make_annotation_zarr=False,
        save_flat_layers_npz=True,
    )

    flat_volume = zarr.open_group(str(art["image_zarr"]), mode="r")["data"]
    flat_layers = np.load(art["flat_layers_npz"])

    maps = compute_extra_enface_maps(
        flat_volume=flat_volume,
        flat_layers=flat_layers,
        reference_key=reference_key,
        slab_offsets=SLAB_OFFSETS,
        interp_x=True,
    )

    maps = {k: v for k, v in maps.items() if k.startswith("slab_mean|")}
    maps = make_enface_dict_isotropic_x(maps, x_scale=2.0, order=1)

    out_dir = OUT_ROOT / vol_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(out_dir / "extra_maps_enface.npz", **maps)

    for k, arr in maps.items():
        tag = k.replace("|", "_").replace("->", "to")
        plt.figure(figsize=(6, 6), dpi=300)
        plt.imshow(arr, cmap="gray")
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(out_dir / f"{tag}.png", bbox_inches="tight", pad_inches=0)
        plt.close()


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    for vol_path in sorted(VOL_ROOT.glob(GLOB)):
        try:
            print(vol_path.name)
            save_one(vol_path)
        except:
            print(f"unable to process {vol_path.name}")


if __name__ == "__main__":
    main()