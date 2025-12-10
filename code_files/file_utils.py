import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import re
from dask import array as da

def load_ss_volume(fp):
    """loads a direct file from numpy if possible too now. Defaults to the shape of 1024 slices, 1536 tall, 512 wide
    """
    if '.npy' in str(fp):
        vol = np.load(fp)
    else:
        flat = np.fromfile(fp, dtype=np.uint16)
        vol = np.rot90(flat.reshape(1024, 1536, 512),k=2,axes=(1,2))             # (Z, height, width)
    return vol

def load_ss_volume2(
    fp: Path,
    z_step: int = 1,
    y_step: int = 1,
    x_step: int = 1,
    mmap: bool = True
) -> np.ndarray:
    """Load (and optionally subsample) a SmartScan volume via mmap."""
    if fp.suffix == '.npy':
        vol = np.load(fp, mmap_mode='r' if mmap else None)
    else:
        shape = (1024, 1536, 512)
        if mmap:
            arr = np.memmap(fp, dtype=np.uint16, mode='r', shape=shape)
        else:
            flat = np.fromfile(fp, dtype=np.uint16)
            arr = flat.reshape(shape)
        vol = np.rot90(arr, k=2, axes=(1, 2))
    # apply subsampling
    return vol[::z_step, ::y_step, ::x_step]

def load_onh_info(fp):
    ONH_info = da.from_zarr(str(fp))                # lazy
    return ONH_info

def downsample_layers(layers,xyz_step=(1,1,1)):
    """dict input of layers"""
    if xyz_step==(1,1,1):
        return layers
    else:
        new_layers: dict[str, np.ndarray] = {}
        for name,layer in layers.items():
            new_layers[name] = layer[::xyz_step[2],::xyz_step[0]]/xyz_step[1]
        return new_layers

def image_to_annotation_path(img_path: Path, ann_root: Path=None) -> Path:
    if ann_root is None:
        ann_root = Path("/Users/matthewhunt/Research/Iowa_Research/Han_AIR/testing_annotations")
    cand1 = ann_root / img_path.with_suffix(".labels.zarr").name
    cand2 = ann_root / img_path.with_suffix(".zarr").name
    return cand1 if cand1.exists() else cand2

    
        
def curves_to_label_vol(
    layers: dict[str, np.ndarray],
    image_height: int = 1536,
    vert_dilation_size: int = 0,   # interpreted as half-thickness (k); total = 2k+1
    names_to_use = ['rpe_raw','rpe_smooth','ilm_raw','ilm_smooth']
) -> np.ndarray:
    """
    layers: dict of {name: (Z, W) curves} with finite Y (row) coords in [0, H-1]
    image_height: full Y dimension of the OCT volume
    vert_dilation_size: half-thickness in pixels; 0 => 1-pixel line
    """
    # Shapes
    Z, W = next(iter(layers.values())).shape
    H = int(image_height)

    # Output dtype: promote if >255 labels
    n_labels = len(layers)
    out_dtype = np.uint16 if n_labels > 255 else np.uint8

    # Allocate output and flatten a view for fast linear writes
    lbl = np.zeros((Z, H, W), dtype=out_dtype)
    lbl_flat = lbl.ravel()  # linear index: (z*H + y)*W + x

    # Precompute grids once
    z_grid = np.arange(Z, dtype=np.int32)[:, None]      # (Z,1)
    x_grid = np.arange(W, dtype=np.int32)[None, :]      # (1,W)
    z_base = (z_grid.astype(np.int64) * (H * W))        # (Z,1), linear base per z

    k = int(vert_dilation_size)
    offsets = np.arange(-k, k + 1, dtype=np.int32)      # (2k+1,)

    for li, (key,curve) in enumerate(layers.items(), start=1):
        # sanitize Y centers
        if key not in names_to_use:
            continue
        y = np.rint(curve).astype(np.int32, copy=False)  # (Z,W)
        valid = np.isfinite(curve)                       # (Z,W)

        # Paint a vertical band (2k+1) around each center; keep_existing rule
        for off in offsets:
            y_off = y + off
            # clip to [0, H-1]
            np.clip(y_off, 0, H - 1, out=y_off)

            # linear indices for all (z,x) at this offset
            lin = (z_base + (y_off.astype(np.int64) * W) + x_grid).ravel()

            # only where the curve is defined
            sel = valid.ravel()
            if not sel.any():
                continue

            idx = lin[sel]
            # write only into background (keep_existing semantics)
            empty = (lbl_flat[idx] == 0)
            if empty.any():
                lbl_flat[idx[empty]] = li + 2

    return lbl

def get_corresponding_layer_path(vol_path,file_suffix = '.npy', dir_suffix = "_processed"):
    """input is a posix path"""
    vol_dir = vol_path.parent
    # Now it's possible we might use the mini versions, but still need to access the same layers names
    processed_dir = vol_dir.with_name(vol_dir.name.strip("_mini").strip("_full") + dir_suffix)
    # layer_path    = processed_dir / (vol_path+file_suffix_prepend).with_suffix(file_suffix).name
    new_filename = f"{vol_path.stem}{file_suffix}"
    # e.g. "data" + "_layers" + ".npy" → "data_layers.npy"

    return processed_dir / new_filename

def load_layers(layer_path,use_mini=False):
    '''loads the layers directly and will adjust as needed for mini, assuming a 1/20 z-stack used'''
    layers = np.load(layer_path)  # shape: (n_slices, W, n_layers)
    # layers = np.load(layer_path, mmap_mode='r')
    if use_mini:
        indices = np.arange(0, layers.shape[0], 20)
        layers = layers[indices,:,:]
    return layers


def crawl_dirs(root_dir, fn):
    """executes and returns a function from walking across all files. Must include file filtering logic in the function"""
    return [
        fn(p)
        for p in sorted(Path(root_dir).rglob("*"))
        if p.is_file()
    ]

def extract_eye_side(filename):
    match = re.search(r'_(OD|OS)_', filename)
    if match:
        return match.group(1)
    return None

# The below was gpt garbage for initial exploration
# 1) configure known image specs: suffix → (shape, dtype)
KNOWN_SPECS = {
    "iris.bin":   ((480, 640), np.uint8),    # iris: 480 rows × 640 cols, 8-bit
    # "lslo.bin":   ((512, 664), np.uint16),   # LSLO fundus: 512 rows × 664 cols, 16-bit
    # add more fixed specs here if you discover them...
}

def infer_square_shape(path, dtype):
    """Assume a square image: infer side = sqrt(n_pixels)."""
    n_bytes = os.path.getsize(path)
    n_pix = n_bytes // np.dtype(dtype).itemsize
    side = int(np.round(np.sqrt(n_pix)))
    print(n_pix)
    print(side)
    if side * side != n_pix:
        raise ValueError(f"Can’t infer square shape for {path}, n_pix={n_pix}")
    return (side, side)

def load_image(path, shape, dtype):
    """Load raw file → NumPy array of given shape & dtype."""
    data = np.fromfile(path, dtype=dtype)
    return data.reshape(shape)

def plot_images_in_folder(folder):
    """Scan for raw files, load & plot each one."""
    patterns = ["*.bin", "*.img"]
    found = []
    for pat in patterns:
        found += glob.glob(os.path.join(folder, pat))
    if not found:
        print("No .bin or .img files found in", folder)
        return

    for fp in sorted(found):
        fn = os.path.basename(fp).split("_")[-1]
        print(fn)
        # Determine spec
        if fn in KNOWN_SPECS:
            shape, dtype = KNOWN_SPECS[fn]
        else:
            # try uint16 by default for .img; adjust if wrong
            dtype = np.uint16
            if "hidef" in fp:
                import pdb; pdb.set_trace()
            try:
                shape = infer_square_shape(fp, dtype)
                print(f"Inferred {shape} for {fn}")
            except ValueError:
                print(f"Skipping {fn}: unknown dims and non-square size")
                continue

        # load & plot
        print(f"trying fp = {fp}")
        img = load_image(fp, shape, dtype)
        plt.figure(figsize=(4,4))
        plt.imshow(img, cmap="gray", aspect="auto")
        plt.title(fn)
        plt.axis("off")

    plt.show()


import os
import numpy as np
import matplotlib.pyplot as plt

def guess_and_plot(path,
                   dtype=np.uint16,
                   depth_guesses=(1024, 1536),
                   dimensions = 2,
                   max_dims=(2048, 2048)):
    """
    Infer and plot plausible 2D or 3D image shapes based on raw file size.
    - For 2D candidates, plots the full image.
    - For 3D candidates, plots the middle slice along the first axis (Z).
    Aspect ratios are preserved (1:1 pixels).
    """
    # Read raw data once
    voxel_size = np.dtype(dtype).itemsize
    total_bytes = os.path.getsize(path)
    total_voxels = total_bytes // voxel_size
    flat = np.fromfile(path, dtype=dtype)

    candidates_2d = []
    candidates_3d = []

    if dimensions == 2:
        # --- 2D guesses ---
        for h in range(64, max_dims[0] + 1, 8):
            if total_voxels % h == 0:
                w = total_voxels // h
                if w <= max_dims[1]:
                    candidates_2d.append((h, w))

        # Plot 2D candidates
        for (h, w) in candidates_2d:
            try:
                img2d = flat.reshape((h, w))
            except ValueError:
                continue
            plt.figure()
            plt.imshow(img2d, cmap="gray")
            plt.title(f"2D candidate: {h}×{w}")
            plt.axis('off')

    # Plot 3D candidates (middle slice)

    if dimensions == 3:
        # --- 3D volume guesses ---
        for depth in depth_guesses:
            for w in range(128, max_dims[1] + 1, 32):
                plane = depth * w
                if plane > 0 and total_voxels % plane == 0:
                    z = total_voxels // plane
                    candidates_3d.append((z, depth, w))

        for (z, depth, w) in candidates_3d:
            try:
                vol3d = flat.reshape((z, depth, w))
            except ValueError:
                continue
            mid_idx = z // 2
            fig,ax = plt.subplots(1,n_slices)
            for i in range(n_slices):
                slice_img = vol3d[mid_idx+i]
                ax[i].imshow(slice_img, cmap="gray", aspect='equal')
            fig.suptitle(f"3D candidate: ({z}, {depth}, {w}) — slice {mid_idx}")

    if not candidates_2d and not candidates_3d:
        print("No plausible shapes found.")
    else:
        plt.show()

    return {
        "2d": candidates_2d,
        "3d": candidates_3d
    }

import os
import itertools
import numpy as np
import matplotlib.pyplot as plt


def guess_and_plot_brute(
        path,
        dtype=np.uint16,
        depth_guesses=(1024, 1536),
        dimensions=2,
        n_slices=5,
        max_dims=(2048, 2048)):
    """
    Infer and plot plausible 2-D or 3-D shapes for a raw .img/.bin file.

    Parameters
    ----------
    path : str
        Path to the raw file.
    dtype : np.dtype
        Voxel data type (default uint16).
    depth_guesses : tuple[int]
        Candidate axial depths for volumes.
    dimensions : {2, 3}
        Whether to search for 2-D or 3-D shapes.
    n_slices : int
        How many consecutive slices to display for 3-D shapes.
    max_dims : (int, int)
        Maximum height/width to consider when brute-forcing.
    """
    voxel_size   = np.dtype(dtype).itemsize
    total_bytes  = os.path.getsize(path)
    total_voxels = total_bytes // voxel_size
    flat         = np.fromfile(path, dtype=dtype)

    candidates_2d, candidates_3d = [], []

    # ------------------------------------------------------------------
    # 2-D search
    # ------------------------------------------------------------------
    if dimensions == 2:
        for h in range(64, max_dims[0] + 1, 32):
            if total_voxels % h:
                continue
            w = total_voxels // h
            if w <= max_dims[1]:
                candidates_2d.append((h, w))

        for (h, w) in candidates_2d:
            try:
                img = flat.reshape((h, w))
            except ValueError:
                continue
            plt.figure()
            plt.imshow(img, cmap="gray")
            plt.title(f"2-D candidate: {h} × {w}")
            plt.axis("off")

    # ------------------------------------------------------------------
    # 3-D search  +  axis-order permutations
    # ------------------------------------------------------------------
    if dimensions == 3:
        for depth in depth_guesses:
            for w in range(128, max_dims[1] + 1, 32):
                plane = depth * w
                if plane == 0 or total_voxels % plane:
                    continue
                z = total_voxels // plane
                candidates_3d.append((z, depth, w))

        # for every candidate shape, try all 6 (Z,Y,X) orderings
        for cand_shape in candidates_3d:
            for perm in itertools.permutations((0, 1, 2)):
                shape_perm = tuple(cand_shape[i] for i in perm)
                try:
                    vol = flat.reshape(shape_perm)
                except ValueError:
                    continue

                stack = shape_perm[0]         # first axis after permutation
                mid   = stack // 2
                start = max(0, mid - n_slices // 2)
                end   = min(stack, start + n_slices)

                fig, axs = plt.subplots(1, end - start,
                                        figsize=(4 * (end - start), 4),
                                        squeeze=False)

                for idx, ax in zip(range(start, end), axs[0]):
                    img = vol[idx]            # axis-0 slice
                    ax.imshow(img, cmap="gray", aspect="equal")
                    ax.set_title(f"slice {idx}")
                    ax.axis("off")

                perm_text = "".join("ZYX"[i] for i in perm)
                fig.suptitle(f"3-D candidate {cand_shape}  |  order {perm_text}")

    if not candidates_2d and not candidates_3d:
        print("No plausible shapes found.")
    else:
        plt.show()

    return {"2d": candidates_2d, "3d": candidates_3d}


