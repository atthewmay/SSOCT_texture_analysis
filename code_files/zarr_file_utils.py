from __future__ import annotations
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))  # adds Han_AIR/ to path
import numpy as np
import code_files.visualization_utils as vu
import code_files.file_utils as fu
from concurrent.futures import ProcessPoolExecutor

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




from pathlib import Path
import zarr
import numcodecs

from code_files import file_utils as fu
from code_files.segmentation_code import segmentation_utility_functions as sfu


def _write_volume_to_zarr_streaming(vol_np: np.ndarray, zarr_path: Path, chunks: tuple[int, int, int], overwrite: bool = True):
    """shoudl pick one of these only"""
    compressor = numcodecs.Blosc(
        cname="lz4",
        clevel=3,
        shuffle=numcodecs.Blosc.BITSHUFFLE,
    )

    Z, Y, X = vol_np.shape
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

    for z in range(Z):
        arr[z, :, :] = vol_np[z]

    zarr.consolidate_metadata(str(zarr_path))
    return zarr_path


def _flatten_one_slice(job):
    z, img, ref_line, lines_dict, fill = job

    flat_img, shift_z, target_z = sfu.flatten_to_path(
        img,
        ref_line,
        fill=fill,
        target_y=None,
    )

    flat_lines = {}
    for name, line in lines_dict.items():
        flat_lines[name] = sfu.warp_line_by_shift(
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
    flattened_artifacts_root: Path=Path('/Volumes/T9/iowa_research/Han_AIR_Dec_2025/flattened_artifacts'),
    z_stride: int = 1,
    overwrite: bool = False,
    make_image_zarr: bool = True,
    make_label_zarr: bool = True,
    save_flat_layers_npz: bool = True,
    fill: float = 0.0,
    vert_dilation_size: int = 2,
):
    """
    Build/reuse reversible flattened artifacts for one volume.

    Outputs beside the source volume:
      *.{flatten_with}.flat_img.zarr
    #   *.{flatten_with}.flat_lbl.zarr
      *.{flatten_with}.flat_layers.npz
      *.{flatten_with}.flat_meta.npz

    flat_meta.npz contains the actual warp info needed for unflattening later.
    """
    vol_path = Path(vol_path)
    vol_name = vol_path.stem
    layers_root = Path(layers_root)

    img_zarr = (flattened_artifacts_root/vol_name).with_suffix(f".{flatten_with}.flat_img.zarr")
    lbl_zarr = (flattened_artifacts_root/vol_name).with_suffix(f".{flatten_with}.flat_lbl.zarr")
    flat_layers_npz = (flattened_artifacts_root/vol_name).with_suffix(f".{flatten_with}.flat_layers.npz")
    flat_meta_npz = (flattened_artifacts_root/vol_name).with_suffix(f".{flatten_with}.flat_meta.npz")

    need_build = overwrite or any(
        not p.exists()
        for p in [flat_meta_npz]
        + ([img_zarr] if make_image_zarr else [])
        + ([lbl_zarr] if make_label_zarr else [])
        + ([flat_layers_npz] if save_flat_layers_npz else [])
    )

    if not need_build:
        print(f"[reuse] flattened artifacts for {vol_path.name} using {flatten_with}. asserting z_stride aligns.")

        assert np.load(flat_meta_npz)['z_stride'] == z_stride
        return {
            "image_zarr": img_zarr if make_image_zarr else None,
            "label_zarr": lbl_zarr if make_label_zarr else None,
            "flat_layers_npz": flat_layers_npz if save_flat_layers_npz else None,
            "flat_meta_npz": flat_meta_npz,
        }

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
        lbl_flat = _paint_labels_from_flat_layers(
            flat_layers,
            image_height=H,
            vert_dilation_size=vert_dilation_size,
        )
        _write_volume_to_zarr_streaming(
            lbl_flat.astype(np.uint8, copy=False),
            lbl_zarr,
            chunks=(1, H, W),
            overwrite=True,
        )
        print(f"[build] labels(flat:{flatten_with}) -> {lbl_zarr}")

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
        "flat_layers_npz": flat_layers_npz if save_flat_layers_npz else None,
        "flat_meta_npz": flat_meta_npz,
    }