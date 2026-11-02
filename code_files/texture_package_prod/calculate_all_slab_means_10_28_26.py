#!/usr/bin/env python3
# REVIEWED
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from tqdm.auto import tqdm

from code_files import file_utils as fu
from code_files.texture_package_prod.texture_enface_utils import slab_stat_map, make_enface_isotropic_x

VOL_ROOT = Path("/Volumes/T9/iowa_research/Han_AIR_Dec_2025/data_volumes/data_all_volumes2")
# LAYERS_ROOT = Path("/Volumes/T9/iowa_research/Han_AIR_Dec_2025/layers_rsync_dir/layers_2026_10_24/layers_2026_10_24")
# OUT_ROOT = Path("/Volumes/T9/iowa_research/Han_AIR_Dec_2025/results/enface_maps_2026_10_24")
LAYERS_ROOT = Path("/Volumes/T9/iowa_research/Han_AIR_Dec_2025/layers_rsync_dir/layers_2026_04_18_merged")
OUT_ROOT = Path("/Volumes/T9/iowa_research/Han_AIR_Dec_2025/results/enface_maps_2026_04_18_merged")
SLABS = [(10, 20)]
WORKERS = 8



def slab_mean_lowmem(vol, rpe, bottom=10, top=20):
    Z, Y, X = vol.shape
    out = np.full((Z, X), np.nan, dtype=np.float32)
    xx = np.arange(X)

    for z in range(Z):
        ref = np.asarray(rpe[z], dtype=np.float32)
        valid_ref = np.isfinite(ref)
        safe_ref = np.where(valid_ref, ref, 0)

        start = np.clip(np.floor(safe_ref - top).astype(int) + 1, 0, Y)
        stop = np.clip(np.ceil(safe_ref - bottom).astype(int), 0, Y)

        sums = np.zeros(X, dtype=np.float64)
        counts = np.zeros(X, dtype=np.int32)

        for offset in range(int(np.max(stop - start))):
            yy = start + offset
            valid = valid_ref & (yy < stop)
            if not valid.any():
                continue
            vals = np.asarray(vol[z, np.clip(yy, 0, Y - 1), xx], dtype=np.float32)
            valid &= np.isfinite(vals)
            sums[valid] += vals[valid]
            counts[valid] += 1

        good = counts > 0
        out[z, good] = sums[good] / counts[good]
        if good.sum() >= 2:
            out[z] = np.interp(xx, xx[good], out[z, good])

    return out

def process(p):
    key = fu.get_algorithm_key_from_filepath(p)
    lp = fu.new_get_corresponding_layer_path(p, layers_root=LAYERS_ROOT)
    with np.load(lp) as layers:
        rpe = layers[key].astype(np.float32)

    vol = fu.load_ss_volume2(p, mmap=True)
    maps = {}
    for bottom, top in SLABS:
        # slab = slab_stat_map(vol, rpe, bottom, top, stat="mean", interp_x=True)
        # slab = slab_mean_lowmem(vol, rpe, bottom, top, stat="mean", interp_x=True)
        slab = slab_mean_lowmem(vol, rpe, bottom, top)
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
                print(f"Failed: {futures[f].name}")
                # raise RuntimeError(f"Failed: {futures[f].name}") from e
    print("Saved to:", OUT_ROOT)


if __name__ == "__main__":
    main()
