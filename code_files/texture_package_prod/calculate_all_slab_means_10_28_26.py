#!/usr/bin/env python3
# REVIEWED
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from tqdm.auto import tqdm

from code_files import file_utils as fu
from code_files.texture_package_prod.texture_enface_utils import slab_stat_map, make_enface_isotropic_x

VOL_ROOT = Path("/Volumes/T9/iowa_research/Han_AIR_Dec_2025/data_volumes/data_all_volumes2")
LAYERS_ROOT = Path("/Volumes/T9/iowa_research/Han_AIR_Dec_2025/layers_rsync_dir/layers_2026_10_24/layers_2026_10_24")
OUT_ROOT = Path("/Volumes/T9/iowa_research/Han_AIR_Dec_2025/results/enface_maps_2026_10_24")
SLABS = [(10, 20)]
WORKERS = 8


def process(p):
    key = fu.get_algorithm_key_from_filepath(p)
    lp = fu.new_get_corresponding_layer_path(p, layers_root=LAYERS_ROOT)
    with np.load(lp) as layers:
        rpe = layers[key].astype(np.float32)

    vol = fu.load_ss_volume2(p, mmap=True)
    maps = {}
    for bottom, top in SLABS:
        slab = slab_stat_map(vol, rpe, bottom, top, stat="mean", interp_x=True)
        maps[f"slab_mean|{bottom}->{top}"] = make_enface_isotropic_x(slab, x_scale=2.0, order=1)

    out = OUT_ROOT / p.stem / "slabs.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **maps)
    return p.name


def main():
    volumes = sorted(p for p in VOL_ROOT.rglob("*.img") if p.stat().st_size)
    print(f"Processing {len(volumes)} volumes")
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(process, p): p for p in volumes}
        for f in tqdm(as_completed(futures), total=len(futures), desc="Generating slabs", unit="vol"):
            try:
                f.result()
            except Exception as e:
                raise RuntimeError(f"Failed: {futures[f].name}") from e
    print("Saved to:", OUT_ROOT)


if __name__ == "__main__":
    main()
