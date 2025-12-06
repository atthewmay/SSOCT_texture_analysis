from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))  # adds Han_AIR/ to path
import numpy as np
import code_files.visualization_utils as vu
import code_files.file_utils as fu


### The following are derived from our napari work. Lot's of good stuff here for zarr and dask to enable silky smooth loading and scrolling

def ensure_image_flat_zarr(vol_path: Path, flatten_with: str, z_stride: int, overwrite: bool = False,dir_suffix: str = '_labels' ) -> Path:
    """
    Build/reuse flattened image Zarr using `flatten_with` as reference layer.
    Output height matches original H so shapes align (Z, H, X).
    """
    flat_zarr = vol_path.with_suffix(f".{flatten_with}.flat.zarr") # REFACTOR: would make part of this string the actual layer dir name and date used to flatten it
    # Should take in as arge the layer dir to use
    print("would refactor here")
    if overwrite or not flat_zarr.exists():
        layer_path = fu.get_corresponding_layer_path(vol_path, file_suffix='_layers.npz', dir_suffix=dir_suffix)
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
        z[k, :, :] = vol_np[k, :, :]

    # Make opens faster (single .zmetadata read)
    zarr.consolidate_metadata(store)
    return zarr_path

