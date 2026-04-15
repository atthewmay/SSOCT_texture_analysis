from __future__ import annotations
from pathlib import Path
import sys

from code_files.segmentation_code.flattening_utility_functions import warp_line_by_shift
sys.path.append(str(Path(__file__).resolve().parents[1]))  # adds Han_AIR/ to path
import numpy as np
import code_files.visualization_utils as vu
import code_files.file_utils as fu
from concurrent.futures import ProcessPoolExecutor

import time
C = fu.load_constants()

### The following are derived from our napari work. Lot's of good stuff here for zarr and dask to enable silky smooth loading and scrolling

# def ensure_image_flat_zarr(vol_path: Path, flatten_with: str, z_stride: int, overwrite: bool = False,dir_suffix: str = '_labels' ) -> Path:
def ensure_image_flat_zarr(vol_path: Path, flatten_with: str, z_stride: int, overwrite: bool = False,layers_root: str=None, dir_suffix: str = None ) -> Path:
    """
    Build/reuse flattened image Zarr using `flatten_with` as reference layer.
    Output height matches original H so shapes align (Z, H, X).
    """
    flat_zarr = vol_path.with_suffix(f".{flatten_with}.flat.zarr") # REFACTOR: would make part of this string the actual layer dir name and date used to flatten it
    # Should take in as arge the layer dir to use
    print("would refactor here")
    if overwrite or not flat_zarr.exists():
        # layer_path = fu.get_corresponding_layer_path(vol_path, file_suffix='_layers.npz', dir_suffix=dir_suffix)
        layer_path = fu.new_get_corresponding_layer_path(vol_path, layers_root=layers_root)
        if not layer_path.exists():
            raise FileNotFoundError(f"Layer file not found for {vol_path}: {layer_path}")

        vol_np = fu.load_ss_volume2(vol_path, z_step=1, y_step=1, x_step=1, mmap=True)  # (Z,H,X)
        H, W = vol_np.shape[1], vol_np.shape[2]
        layers = np.load(layer_path)

        # Flatten volume to keep height H so downstream shapes match
        flat_vol = vu.flatten_volume(vol_np, layers[flatten_with], output_height=int(H))
        if z_stride > 1:
            flat_vol = flat_vol[::z_stride]
        write_volume_to_zarr_streaming(flat_vol.astype(vol_np.dtype, copy=False), flat_zarr, (1, H, W))
        print(f"[build] image(flat:{flatten_with}) → {flat_zarr}")
    else:
        print(f"[reuse] image(flat:{flatten_with}) → {flat_zarr}")
    return flat_zarr


def ensure_labels_flat_zarr(vol_path: Path, flatten_with: str, z_stride: int, overwrite: bool = False,dir_suffix: str = '_labels') -> Path:
    """
    Build/reuse flattened labels Zarr by flattening curves with the same offsets
    (via vu.flatten_other_curves) and repainting labels at height H.
    """
    layer_path = fu.get_corresponding_layer_path(vol_path, file_suffix='_layers.npz', dir_suffix=dir_suffix)
    if not layer_path.exists():
        raise FileNotFoundError(f"Layer file not found for {vol_path}: {layer_path}")

    labels_flat_zarr = layer_path.with_suffix(f".{flatten_with}.flat.zarr")
    if overwrite or not labels_flat_zarr.exists():
        vol_np = fu.load_ss_volume2(vol_path, z_step=1, y_step=1, x_step=1, mmap=True)
        H, W = vol_np.shape[1], vol_np.shape[2]
        layers = np.load(layer_path)
        layers = fu.downsample_layers(layers, (1, 1, 1))

        flat_layers = vu.flatten_other_curves(layers, ref_layer=flatten_with, output_height=int(H))
        lbl_flat = fu.curves_to_label_vol(flat_layers, image_height=H, vert_dilation_size=2)
        if z_stride > 1:
            lbl_flat = lbl_flat[::z_stride]
        write_volume_to_zarr_streaming(lbl_flat.astype(np.uint8, copy=False), labels_flat_zarr, (1, H, W))
        print(f"[build] labels(flat:{flatten_with}) → {labels_flat_zarr}")
    else:
        print(f"[reuse] labels(flat:{flatten_with}) → {labels_flat_zarr}")
    return labels_flat_zarr

def write_volume_to_zarr_streaming(vol_np: np.ndarray, zarr_path: Path, chunks: tuple[int,int,int]):
    import zarr, numcodecs, numpy as np

    # Faster interactive browsing than zstd; still compressed.
    compressor = numcodecs.Blosc(cname="lz4", clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE)

    Z, Y, X = vol_np.shape
    store = zarr.DirectoryStore(str(zarr_path))
    z = zarr.open(
        store, mode="w",
        shape=(Z, Y, X), chunks=chunks, dtype=vol_np.dtype, compressor=compressor
    )

    # Stream slice-by-slice (bounded RAM)
    for k in range(Z):
        z[k, :, :] = vol_np[k, :, :] #note i think this assumes that vol_np is opned w/ mmap

    # Make opens faster (single .zmetadata read)
    zarr.consolidate_metadata(store)
    return zarr_path


def ensure_image_zarr(vol_path: Path, z_stride: int,overwrite: bool) -> Path:
    """Create/reuse an image Zarr next to vol_path, thinning Z if requested."""
    zarr_path = vol_path.with_suffix(".zarr")
    if not zarr_path.exists() or overwrite == True:
        vol_np = fu.load_ss_volume2(vol_path, z_step=1, y_step=1, x_step=1, mmap=True) # slightly less than optimal to load entire z-dim, but only occurs 1x
        H, W = vol_np.shape[1], vol_np.shape[2]
        chunks = (1, H, W)  # full B-scan per chunk
        write_volume_to_zarr_streaming(vol_np[::max(1, z_stride)], zarr_path, chunks)
        print(f"[build] image → {zarr_path}")
    else:
        print(f"[reuse] image → {zarr_path}")
    return zarr_path


def write_labels_group_to_zarr_streaming(
    vols: dict[str, np.ndarray],
    zarr_path: Path,
    chunks: tuple[int, int, int],
):
    """
    vols: {name: (Z,H,W) uint8/uint16 label volumes}
    writes to a zarr group at `zarr_path`, one dataset per key.
    """
    import zarr, numcodecs, numpy as np

    compressor = numcodecs.Blosc(
        cname="lz4",
        clevel=3,
        shuffle=numcodecs.Blosc.BITSHUFFLE,
    )

    try: # handlign multiple envs
        root = zarr.open_group(str(zarr_path), mode="w", zarr_format=2)
        for name, v in vols.items():
            assert v.ndim == 3, (name, v.shape)
            Z, H, W = v.shape
            root.create_array(
                name=name,
                shape=(Z, H, W),
                chunks=chunks,
                dtype=v.dtype,
                compressor=compressor,
                overwrite=True,
                fill_value=0,
            )
    except TypeError:
        root = zarr.open_group(str(zarr_path), mode="w")
        for name, v in vols.items():
            assert v.ndim == 3, (name, v.shape)
            Z, H, W = v.shape
            root.create_dataset(
                name=name,
                shape=(Z, H, W),
                chunks=chunks,
                dtype=v.dtype,
                compressor=compressor,
                overwrite=True,
                fill_value=0,
            )

    any_vol = next(iter(vols.values()))
    Z = any_vol.shape[0]
    for k in range(Z):
        for name, v in vols.items():
            root[name][k, :, :] = v[k, :, :]

    zarr.consolidate_metadata(str(zarr_path))
    return zarr_path

def ensure_labels_zarr(vol_path: Path, z_stride: int,overwrite: bool,layers_root: str ) -> Path:
    """Create/reuse a labels Zarr from the *_layers.npz file aligned to vol_path.
    if supplied a labels_dir
    1/26/26: now accepting a zarr folder with subfoldersing during the refactorj
    """
    # layer_path = fu.get_corresponding_layer_path(vol_path, file_suffix='_layers.npz', dir_suffix=dir_suffix)
    # layer_path = fu.get_corresponding_layer_path(vol_path, file_suffix='', dir_suffix=dir_suffix) # For january script, just keep name for simplicity
    layer_path = fu.new_get_corresponding_layer_path(vol_path,layers_root=layers_root) # For january script, just keep name for simplicity
    if not layer_path.exists():
        print(f"Layer file not found for {vol_path}. For now will simply return")
        return None
        raise FileNotFoundError(f"Layer file not found for {vol_path}: {layer_path}")
    labels_zarr = layer_path.with_suffix(".zarr") # now a dir with group

    if not labels_zarr.exists() or overwrite == True:
        # Load image *shape* via memmap to get H,W without big RAM
        vol_np = fu.load_ss_volume2(vol_path, z_step=1, y_step=1, x_step=1, mmap=True) #right, doesn't really load it
        H, W = vol_np.shape[1], vol_np.shape[2]

        layers = np.load(layer_path)                      # (Z, W, n_layers) dict-like npz or array
        layers = fu.downsample_layers(layers, (1, 1, 1))  # keep XY native for alignment

        vols = _build_label_set_vols(layers,H,1,z_stride=z_stride)

        write_labels_group_to_zarr_streaming(vols, labels_zarr, chunks=(1, H, W))

        print(f"[build] labels → {labels_zarr}")
    else:
        print(f"[reuse] labels → {labels_zarr}")
    return labels_zarr


# Might it just be simpler to modify this function s.t. w/ triggering from a string name. I think if the string starts with "slab_..." then we invoke just a modificaiton in the layer s.t. we draw from the dictionary to find the correct name, make like two temp layers w/ the proper offset, and pass those into fu.curves_to_label_vol(. Why not just do this?
def _parse_layer_spec(spec: str):
    """
    Accept either:
      - 'rpe_smooth'
      - 'rpe_smooth|10:20'
      - 'rpe_smooth|-20:-10'
    """
    spec = spec.strip()
    if "|" not in spec:
        return "curve", spec, None

    base, offs = spec.split("|", 1)
    a, b = offs.split(":", 1)
    return "slab", base.strip(), (int(a), int(b))


def _build_slab_vol_from_curve(curve, image_height, offsets):
    """
    curve: (Z, W)
    offsets are relative to the curve in image-row coordinates.
    Positive = deeper/below, negative = above.
    Returns a binary (Z, H, W) volume with 1 inside the slab.
    """
    t1 = time.time()
    curve = np.asarray(curve, dtype=np.float32)
    Z, W = curve.shape
    H = int(image_height)
    out = np.zeros((Z, H, W), dtype=np.uint8)

    off0, off1 = offsets
    x_all = np.arange(W, dtype=np.int32)

    for z in range(Z):
        row = curve[z]
        valid = np.isfinite(row)
        if not np.any(valid):
            continue

        x = x_all[valid]
        y = np.rint(row[valid]).astype(np.int32)

        y0 = np.clip(y - off0, 0, H - 1)
        y1 = np.clip(y - off1, 0, H - 1)

        y_lo = np.minimum(y0, y1)
        y_hi = np.maximum(y0, y1)

        for xi, lo, hi in zip(x, y_lo, y_hi):
            out[z, lo:hi + 1, xi] = 1

    
    print(f"built a slab from curves in {time.time()-t1}")

    return out

def _build_label_set_vols(layers, image_height, vert_dilation_size=1, z_stride=1):
    LABEL_SETS = {
        'basics': [
            "hypersmoother_path",
            "rpe_smooth",
            "ilm_raw",
            "ilm_smooth",
            # example slab:
            # "rpe_smooth|10:20",
        ],
        'ILM': ["ilm_raw", "ilm_smooth"],
        'two_layer_original': [
            'original_method_y1_vertical_shifted',
            'original_method_y2_vertical_shifted',
        ],
        'two_layer_choroidal': [
            'choroidal_method_y1_vertical_shifted',
            'choroidal_method_y2_vertical_shifted',
        ],
        'two_layer_EZ': [
            'EZ_method_y1_vertical_shifted',
            'EZ_method_y2_vertical_shifted',
        ],
        'all_methods_RPE': [
            'original_method_y2_vertical_shifted',
            'choroidal_method_y1_vertical_shifted',
            'EZ_method_y2_vertical_shifted',
        ],

        # cleaner usage as separate toggleable sets:
        'rpe_slab_10_20_original': ['original_method_y2_vertical_shifted|10:20'],
        'rpe_slab_10_20_EZ': ['EZ_method_y2_vertical_shifted|10:20'],
        'rpe_slab_10_20_choroidal': ['choroidal_method_y1_vertical_shifted|10:20'],
    }

    class _LayerShim:
        def __init__(self, d):
            self._d = d
            self.files = list(d.keys())

        def __getitem__(self, key):
            return self._d[key]

    layer_shim = _LayerShim(layers)

    for example in layer_shim._d.values():
        if np.asarray(example).ndim == 2:
            break
    else:
        raise ValueError("No 2D layer arrays found")
    Z, W = np.asarray(example).shape
    H = int(image_height)

    vols = {}
    for set_name, specs in LABEL_SETS.items():
        vol = np.zeros((Z, H, W), dtype=np.uint16)
        label_idx = 1

        for spec in specs:
            kind, base_name, offsets = _parse_layer_spec(spec)

            if base_name not in layer_shim._d:
                continue

            if kind == "curve":
                tmp = fu.curves_to_label_vol(
                    layer_shim,
                    image_height=image_height,
                    vert_dilation_size=vert_dilation_size,
                    names_to_use=[base_name],
                )
                mask = tmp > 0

            elif kind == "slab":
                tmp = _build_slab_vol_from_curve(
                    layer_shim[base_name],
                    image_height=image_height,
                    offsets=offsets,
                )
                mask = tmp > 0

            else:
                raise ValueError(f"Unknown layer spec kind: {kind}")

            vol[(vol == 0) & mask] = label_idx
            label_idx += 1

        if label_idx == 1:
            continue

        if z_stride > 1:
            vol = vol[::z_stride]

        out_dtype = np.uint8 if (label_idx - 1) <= 255 else np.uint16
        vols[set_name] = vol.astype(out_dtype, copy=False)

    return vols


# def _build_label_set_vols(layers, image_height, vert_dilation_size=1, z_stride=1):
#     LABEL_SETS = {
#         'basics': ["hypersmoother_path", "rpe_smooth", "ilm_raw", "ilm_smooth"],
#         'ILM': ["ilm_raw", "ilm_smooth"],
#         'two_layer_original': [
#             'original_method_y1_vertical_shifted',
#             'original_method_y2_vertical_shifted',
#         ],
#         'two_layer_choroidal': [
#             'choroidal_method_y1_vertical_shifted',
#             'choroidal_method_y2_vertical_shifted',
#         ],
#         'two_layer_EZ': [
#             'EZ_method_y1_vertical_shifted',
#             'EZ_method_y2_vertical_shifted',
#         ],
#         'all_methods_RPE': [
#             'original_method_y2_vertical_shifted',
#             'choroidal_method_y1_vertical_shifted',
#             'EZ_method_y2_vertical_shifted',
#         ],
        
#     }

#     class _LayerShim:
#         def __init__(self, d):
#             self._d = d
#             self.files = list(d.keys())

#         def __getitem__(self, key):
#             return self._d[key]

#     layer_shim = _LayerShim(layers)


#     vols = {}
#     for set_name, names in LABEL_SETS.items():
#         names = [n for n in names if n in layer_shim._d.keys()]
#         if not names:
#             continue
#         v = fu.curves_to_label_vol(
#             layer_shim,
#             image_height=image_height,
#             vert_dilation_size=vert_dilation_size,
#             names_to_use=names,
#         )
#         if z_stride > 1:
#             v = v[::z_stride]
#         vols[set_name] = v.astype(np.uint8, copy=False)
#     return vols



# def get_texture_zarr_path(vol_path: Path, texture_zarr_root: str | Path | None):
#     if texture_zarr_root is None:
#         return None
#     texture_zarr_root = Path(texture_zarr_root)
#     return texture_zarr_root / vol_path.stem / "texture_bscan_maps.zarr"


# from pathlib import Path


def list_texture_run_dirs(vol_path: Path, texture_zarr_root: str | Path | None):
    if texture_zarr_root is None:
        return []

    base = Path(texture_zarr_root) / vol_path.stem
    if not base.exists():
        return []

    run_dirs = []
    for p in sorted(base.iterdir()):
        if not p.is_dir():
            continue
        if (p / "texture_bscan_maps.zarr").exists() or (p / "texture_bscan_maps_compact.zarr").exists():
            run_dirs.append(p)

    return run_dirs


def get_texture_zarr_path(
    vol_path: Path,
    texture_zarr_root: str | Path | None,
    texture_run: str | None = None,
):
    if texture_zarr_root is None:
        return None

    base = Path(texture_zarr_root) / vol_path.stem
    print(f"base = {base}")

    # legacy non-sweep layout
    legacy = base / "texture_bscan_maps.zarr"
    if legacy.exists():
        return legacy

    # explicit run requested
    if texture_run is not None:
        p = base / texture_run / "texture_bscan_maps.zarr"
        return p if p.exists() else None

    # auto-resolve only if exactly one run exists
    run_dirs = list_texture_run_dirs(vol_path, texture_zarr_root)
    print(f"run_dirs = {run_dirs}")
    if len(run_dirs) == 1:
        if (run_dirs[0] / "texture_bscan_maps.zarr").exists():
            print(f"Returning full-size zarr at {run_dirs[0] / 'texture_bscan_maps.zarr'}")
            return run_dirs[0] / "texture_bscan_maps.zarr"
        elif (run_dirs[0] / "texture_bscan_maps_compact.zarr").exists():
            print(f"Returning compact-size zarr at {run_dirs[0] / 'texture_bscan_maps_compact.zarr'}")
            return run_dirs[0] / "texture_bscan_maps_compact.zarr"

    if len(run_dirs) == 0:
        return None

    raise ValueError(
        f"Multiple texture runs found for {vol_path.stem}: "
        f"{[p.name for p in run_dirs]}. "
        f"Pass texture_run=..."
    )



from pathlib import Path
import zarr
import numcodecs

from code_files import file_utils as fu
from code_files.segmentation_code.flattening_utility_functions import flatten_to_path


def _write_volume_to_zarr_streaming(vol_np: np.ndarray, zarr_path: Path, chunks: tuple[int, int, int], overwrite: bool = True):
    """shoudl pick one of these only"""
    compressor = numcodecs.Blosc(
        cname="lz4",
        clevel=3,
        shuffle=numcodecs.Blosc.BITSHUFFLE,
    )

    Z, Y, X = vol_np.shape
    # root = zarr.open_group(str(zarr_path), mode="w" if overwrite else "a", zarr_format=2)
    try: # so ugly. An issue with zarr versions. unfortunately not easily fixable rn
        root = zarr.open_group(str(zarr_path), mode="w" if overwrite else "a", zarr_format=2)
        arr = root.create_array(
            name="data",
            shape=(Z, Y, X),
            chunks=chunks,
            dtype=vol_np.dtype,
            compressor=compressor,
            overwrite=overwrite,
            fill_value=0,
        )


    except TypeError:
        root = zarr.open_group(str(zarr_path), mode="w" if overwrite else "a")
        arr = root.create_dataset(
            name="data",
            shape=(Z, Y, X),
            chunks=chunks,
            dtype=vol_np.dtype,
            compressor=compressor,
            overwrite=overwrite,
            fill_value=0,
        )


    for z in range(Z):
        arr[z, :, :] = vol_np[z]

    zarr.consolidate_metadata(str(zarr_path))
    return zarr_path


def _flatten_one_slice(job):
    z, img, ref_line, lines_dict, fill,target_y_global = job

    flat_img, shift_z, target_z = flatten_to_path(
        img,
        ref_line,
        fill=fill,
        target_y=target_y_global,
    )

    flat_lines = {}
    for name, line in lines_dict.items():
        flat_lines[name] = warp_line_by_shift(
            line,
            shift_z,
            direction="to_flat",
        ).astype(np.float32, copy=False)

    return z, flat_img.astype(np.float32, copy=False), shift_z.astype(np.float32, copy=False), float(target_z), flat_lines


def _flatten_volume_and_layers_to_ref(
    vol_np: np.ndarray,
    layers: dict[str, np.ndarray],
    flatten_with: str,
    fill: float = 0.0,
    target_y_global: int = None,
    max_workers: int=8,
):
    if flatten_with not in layers:
        raise KeyError(f"{flatten_with!r} not found in layers: {list(layers)}")

    Z, H, W = vol_np.shape
    ref = np.asarray(layers[flatten_with], dtype=np.float32)
    if ref.shape != (Z, W):
        raise ValueError(f"Reference layer must have shape {(Z, W)}, got {ref.shape}")

    vol_flat = np.empty((Z, H, W), dtype=np.float32)
    shift_y_full = np.empty((Z, W), dtype=np.float32)
    target_y = np.empty(Z, dtype=np.float32)

    flat_layers = {}
    for name, arr in layers.items():
        if name == "z":
            continue
        arr = np.asarray(arr)
        if arr.shape == (Z, W):
            flat_layers[name] = np.empty((Z, W), dtype=np.float32)

    jobs = [
        (
            z,
            vol_np[z],
            ref[z],
            {name: layers[name][z] for name in flat_layers},
            fill,
            target_y_global
        )
        for z in range(Z)
    ]

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        results = list(ex.map(_flatten_one_slice, jobs, chunksize=8))

    for z, flat_img, shift_z, target_z_z, flat_lines_z in results:
        vol_flat[z] = flat_img
        shift_y_full[z] = shift_z
        target_y[z] = target_z_z

        for name in flat_layers:
            flat_layers[name][z] = flat_lines_z[name]

    meta = {
        "flatten_with": flatten_with,
        "shift_y_full": shift_y_full,
        "target_y": target_y,
        "fill": float(fill),
    }
    return vol_flat, flat_layers, meta


def _paint_labels_from_flat_layers(
    flat_layers: dict[str, np.ndarray],
    image_height: int,
    vert_dilation_size: int = 2,
) -> np.ndarray:
    """
    curves_to_label_vol currently expects an npz-like object with .files.
    Wrap the dict in a tiny shim so we can reuse it unchanged.
    """
    class _LayerShim:
        def __init__(self, d):
            self._d = d
            self.files = list(d.keys())

        def __getitem__(self, key):
            return self._d[key]

    layer_shim = _LayerShim(flat_layers)
    names_to_use = list(flat_layers.keys())

    return fu.curves_to_label_vol(
        layer_shim,
        image_height=image_height,
        vert_dilation_size=vert_dilation_size,
        names_to_use=names_to_use,
    )


def ensure_flattened_artifacts(
    vol_path: Path,
    flatten_with: str,
    *,
    layers_root: str | Path,
    annotation_root: str | Path = None,
    make_annotation_zarr: bool=True,
    flattened_artifacts_root: Path=Path(C['flattened_artifacts_root']),
    z_stride: int = 1,
    overwrite: bool = False,
    make_image_zarr: bool = True,
    make_label_zarr: bool = True,
    save_flat_layers_npz: bool = True,
    fill: float = 0.0,
    vert_dilation_size: int = 2,
    include_texture_zarr: bool = False,
    texture_zarr_root: str | Path | None = None,
):
    """
    Build/reuse reversible flattened artifacts for one volume.

    Outputs beside the source volume:
      *.{flatten_with}.flat_img.zarr
    #   *.{flatten_with}.flat_lbl.zarr
      *.{flatten_with}.flat_layers.npz
      *.{flatten_with}.flat_meta.npz

    flat_meta.npz contains the actual warp info needed for unflattening later.

    What is a little messy is that this has several passthroughs. the function itself exists to both 1. enforce flattening of the flatten-requiring structure (img andlayers, along w/ creating the flat layer-labels vol), 
    and 2. organize the artifacts into single API. Thus there are passthroughs for annotations and textures. Fine for now. 
    """
    vol_path = Path(vol_path)
    vol_name = vol_path.stem
    layers_root = Path(layers_root)

    img_zarr = (flattened_artifacts_root/vol_name).with_suffix(f".{flatten_with}.flat_img.zarr")
    lbl_zarr = (flattened_artifacts_root/vol_name).with_suffix(f".{flatten_with}.flat_lbl.zarr")
    flat_layers_npz = (flattened_artifacts_root/vol_name).with_suffix(f".{flatten_with}.flat_layers.npz")
    flat_meta_npz = (flattened_artifacts_root/vol_name).with_suffix(f".{flatten_with}.flat_meta.npz")

    annotation_zarr = None
    if make_annotation_zarr and annotation_root is not None:
        annotation_path = Path(annotation_root) / vol_path.with_suffix(".labels.zarr").name
        if annotation_path.exists():
            annotation_zarr = ensure_image_zarr(
                annotation_path,
                z_stride=z_stride,
                overwrite=False,
            )
        else:
            print(f"[info] annotation file not found for {vol_path.name}: {annotation_path}")

    texture_zarr = None
    if include_texture_zarr:
        cand = get_texture_zarr_path(vol_path, texture_zarr_root)
        if cand is not None and cand.exists():
            texture_zarr = cand
        else:
            print(f"[info] texture zarr not found for {vol_path.name}: {cand}")


    need_build = overwrite or any(
        not p.exists()
        for p in [flat_meta_npz]
        + ([img_zarr] if make_image_zarr else [])
        + ([lbl_zarr] if make_label_zarr else [])
        + ([flat_layers_npz] if save_flat_layers_npz else [])
    )

    if not need_build:
        print(f"[reuse] flattened artifacts for {vol_path.name} using {flatten_with}. not asserting z_stride aligns.")

        # assert np.load(flat_meta_npz)['z_stride'] == z_stride

        return {
            "image_zarr": img_zarr if make_image_zarr else None,
            "label_zarr": lbl_zarr if make_label_zarr else None,
            "annotation_zarr": annotation_zarr,
            "texture_zarr": texture_zarr,
            "flat_layers_npz": flat_layers_npz if save_flat_layers_npz else None,
            "flat_meta_npz": flat_meta_npz,
        }



    print(f"[building] flattened artifacts for {vol_path.name} using {flatten_with}. asserting z_stride aligns.")
    layer_path = fu.new_get_corresponding_layer_path(vol_path, layers_root=layers_root)
    if not layer_path.exists():
        raise FileNotFoundError(f"Layer file not found for {vol_path}: {layer_path}")

    vol_np = fu.load_ss_volume2(vol_path, z_step=1, y_step=1, x_step=1, mmap=True)  # (Z,H,X)
    Z, H, W = vol_np.shape
    layers_npz = np.load(layer_path)

    layers = {k: layers_npz[k] for k in layers_npz.files if k != "z"}
    vol_flat, flat_layers, flat_meta = _flatten_volume_and_layers_to_ref(
        vol_np=vol_np,
        layers=layers,
        flatten_with=flatten_with,
        fill=fill,
        target_y_global=H//2,
    )

    if z_stride > 1:
        vol_flat = vol_flat[::z_stride]
        for k in list(flat_layers.keys()):
            flat_layers[k] = flat_layers[k][::z_stride]
        flat_meta["shift_y_full"] = flat_meta["shift_y_full"][::z_stride]
        flat_meta["target_y"] = flat_meta["target_y"][::z_stride]

    if make_image_zarr:
        _write_volume_to_zarr_streaming(
            vol_flat.astype(vol_np.dtype, copy=False),
            img_zarr,
            chunks=(1, H, W),
            overwrite=True,
        )
        print(f"[build] image(flat:{flatten_with}) -> {img_zarr}")

    if make_label_zarr:
        vols = _build_label_set_vols(
            flat_layers,
            image_height=H,
            vert_dilation_size=vert_dilation_size,
            z_stride=1,
        )
        write_labels_group_to_zarr_streaming(vols, lbl_zarr, chunks=(1, H, W))

    if save_flat_layers_npz:
        np.savez_compressed(flat_layers_npz, **flat_layers)
        print(f"[build] flat layers -> {flat_layers_npz}")

    np.savez_compressed(
        flat_meta_npz,
        flatten_with=np.array(flat_meta["flatten_with"]),
        shift_y_full=flat_meta["shift_y_full"].astype(np.float32),
        target_y=flat_meta["target_y"].astype(np.float32),
        fill=np.array(flat_meta["fill"], dtype=np.float32),
        z_stride=z_stride,
    )
    print(f"[build] flat meta -> {flat_meta_npz}")


    return {
        "image_zarr": img_zarr if make_image_zarr else None,
        "label_zarr": lbl_zarr if make_label_zarr else None,
        "annotation_zarr": annotation_zarr,
        "texture_zarr": texture_zarr,
        "flat_layers_npz": flat_layers_npz if save_flat_layers_npz else None,
        "flat_meta_npz": flat_meta_npz,
    }

def ensure_nonflat_artifacts(
    vol_path: Path,
    *,
    layers_root: str | Path,
    annotation_root: str | Path | None = None,
    z_stride: int = 1,
    overwrite: bool = False,
    make_image_zarr: bool = True,
    make_label_zarr: bool = True,
    make_annotation_zarr: bool = True,
):
    """
    Build/reuse native (non-flattened) artifacts for one volume.

    Returns paths for:
      - raw image zarr
      - raw label zarr-group
      - annotation zarr (ONH/fovea), if present

    Notes
    -----
    This mirrors the current napari viewer behavior for annotations:
      * annotation path is resolved as annotation_root / <vol_stem>.labels.zarr
      * annotation zarr is always reused with overwrite=False
      * z-stride is handled later at load time, not in the cached annotation path
    """
    vol_path = Path(vol_path)
    layers_root = Path(layers_root)

    if annotation_root is None:
        annotation_root = Path(fu.C["annotation_root"])
    else:
        annotation_root = Path(annotation_root)

    out = {
        "image_zarr": None,
        "label_zarr": None,
        "annotation_zarr": None,
        "texture_zarr": None,
    }

    if make_image_zarr:
        out["image_zarr"] = ensure_image_zarr(
            vol_path,
            z_stride=z_stride,
            overwrite=False, # Hard set, will likely want to change later
        )

    if make_label_zarr:
        layer_path = fu.new_get_corresponding_layer_path(vol_path, layers_root=layers_root)
        if not layer_path.exists():
            raise FileNotFoundError(f"Layer file not found for {vol_path}: {layer_path}")

        out["label_zarr"] = ensure_labels_zarr(
            vol_path,
            z_stride=z_stride,
            overwrite=overwrite,
            layers_root=layers_root,
        )

    if make_annotation_zarr:
        annotation_path = Path(annotation_root) / vol_path.with_suffix(".labels.zarr").name
        if annotation_path.exists():
            out["annotation_zarr"] = ensure_image_zarr(
                annotation_path,
                z_stride=z_stride,
                overwrite=False,
            )
        else:
            print(f"[info] annotation file not found for {vol_path.name}: {annotation_path}")

    return out