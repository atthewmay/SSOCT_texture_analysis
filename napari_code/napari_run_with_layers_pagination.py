import zarr
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))  # adds Han_AIR/ to path
import napari
import numpy as np

#!/usr/bin/env python3
import argparse
import code_files.file_utils as fu
import code_files.visualization_utils as vu

from segmentation_button import add_segmentation_button

# Multithreading hack for older python
import sys, importlib.util
main = sys.modules['__main__']
if not hasattr(main, '__spec__'):
    main.__spec__ = importlib.util.spec_from_loader('__main__', loader=None)


def parse_args():
    p = argparse.ArgumentParser(
        description="Load an OCT volume and its corresponding layer predictions"
    )

    p.add_argument('--flatten_with',
                   default=None,
                   type=str,
                   help="choose a layer to flatten according to")

    p.add_argument('--overwrite_files',
            action="store_true",
            help="overwrite all zaars"
    )

    p.add_argument('--overwrite_labels',
            action="store_true",
            help="overwrite just the label zaars"
    )
                   
    p.add_argument('--regenerate_all_volumes_only',
            action="store_true",
            help="overwrite all zaars without opening the gui. much more convenient! Must have overwrite_files set true tho"
    )

    p.add_argument(
            "--write_zarr",
            action="store_true",
            help="Write a Zarr next to the input and view it lazily."
    )
    p.add_argument(
            "--z_stride",
            type=int,
            default=1,
            help="Thin Z by this stride when writing Zarr (keeps XY native)."
    )
    p.add_argument(
        "--vol_dir",
        type=Path,
        default=None,
        help="If set, load up to 12 volumes from this directory (ignores vol_path)."
    )
    p.add_argument(
        "--glob",
        type=str,
        default="*.npy",
        help="Glob pattern inside --vol_dir to find volumes (default: *.npy)."
    )

    p.add_argument(
        "--labels_dir_suffix",
        type=str,
        default="_layers",
        help="Glob pattern inside --vol_dir to find volumes (default: *.npy)."
    )



    return p.parse_args()

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

def ensure_labels_zarr(vol_path: Path, z_stride: int,overwrite: bool,dir_suffix: str ) -> Path:
    """Create/reuse a labels Zarr from the *_layers.npz file aligned to vol_path.
    if supplied a labels_dir"""
    layer_path = fu.get_corresponding_layer_path(vol_path, file_suffix='_layers.npz', dir_suffix=dir_suffix)
    if not layer_path.exists():
        raise FileNotFoundError(f"Layer file not found for {vol_path}: {layer_path}")
    labels_zarr = layer_path.with_suffix(".zarr")

    if not labels_zarr.exists() or overwrite == True:
        # Load image *shape* via memmap to get H,W without big RAM
        vol_np = fu.load_ss_volume2(vol_path, z_step=1, y_step=1, x_step=1, mmap=True) #right, doesn't really load it
        H, W = vol_np.shape[1], vol_np.shape[2]

        layers = np.load(layer_path)                      # (Z, W, n_layers) dict-like npz or array
        layers = fu.downsample_layers(layers, (1, 1, 1))  # keep XY native for alignment

        lbl_vol = fu.curves_to_label_vol(
            layers,
            image_height=H,
            vert_dilation_size=3,
            names_to_use=['ilm_smooth','rpe_smooth']
        )
        if z_stride > 1:
            lbl_vol = lbl_vol[::z_stride]

        write_volume_to_zarr_streaming(lbl_vol.astype(np.uint8, copy=False), labels_zarr, (1, H, W))
        print(f"[build] labels → {labels_zarr}")
    else:
        print(f"[reuse] labels → {labels_zarr}")
    return labels_zarr

def ensure_image_flat_zarr(vol_path: Path, flatten_with: str, z_stride: int, overwrite: bool = False,dir_suffix: str = '_labels' ) -> Path:
    """
    Build/reuse flattened image Zarr using `flatten_with` as reference layer.
    Output height matches original H so shapes align (Z, H, X).
    """
    flat_zarr = vol_path.with_suffix(f".{flatten_with}.flat.zarr")
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


import time
from dask import array as da
def from_zarr_fresh(path: str):
    za = zarr.open_array(path, mode="r", cache_metadata=False)
    # include mtime so unchanged files still benefit from caching
    mtime = Path(path).stat().st_mtime_ns
    return da.from_zarr(za, name=f"from_zarr-{mtime}-{time.time_ns()}")

def load_one_volume(vp: Path, z_stride: int, overwrite: bool, flatten_with: str | None,dir_suffix:str = '_layers'):
    """
    Return (img, lbl, name) for a single volume:
      - If flatten_with is None: img.shape = (Z, Y, X)
      - Else:                    img.shape = (C=2, Z, Y, X) with C=[raw, flat]
    All arrays are dask-backed from on-disk Zarrs (no full load).
    """
    # from dask import array as da

    # ensure raw caches
    img_z = ensure_image_zarr(vp, z_stride, overwrite)
    lbl_z = ensure_labels_zarr(vp, z_stride, OVERWRITE_LABELS,dir_suffix=dir_suffix)
    img_raw = da.from_zarr(str(img_z))     # (Z,Y,X)
    lbl_raw = da.from_zarr(str(lbl_z))


    annotation_path = Path('/Users/matthewhunt/Research/Iowa_Research/Han_AIR/testing_annotations') / vp.with_suffix('.labels.zarr').name 
    annotation_img = None
    if annotation_path.exists():
        annotation_img = ensure_image_zarr(annotation_path, z_stride, overwrite=False) # overwrite hard-set to false bc you don't ever re-compute the annotations here! 
        # the annotations are made with the separate python file
        # annotation_img = da.from_zarr(str(annotation_img))
        annotation_img = from_zarr_fresh(str(annotation_img))

    if not flatten_with:
        # Rechunk for smooth Z scrolling: (1,H,W)
        img = img_raw.rechunk((1, img_raw.shape[-2], img_raw.shape[-1]))
        lbl = lbl_raw.rechunk((1, lbl_raw.shape[-2], lbl_raw.shape[-1]))
        return img, lbl, annotation_img,vp.stem

    # ensure flat caches
    img_f = ensure_image_flat_zarr(vp, flatten_with, z_stride, overwrite,dir_suffix=dir_suffix)
    lbl_f = ensure_labels_flat_zarr(vp, flatten_with, z_stride, OVERWRITE_LABELS,dir_suffix=dir_suffix)
    img_flat = da.from_zarr(str(img_f))    # (Z,Y,X)
    lbl_flat = da.from_zarr(str(lbl_f))

    # Align Z to the smaller one (defensive)
    common_Z = min(img_raw.shape[0], img_flat.shape[0])
    img_raw = img_raw[:common_Z]; lbl_raw = lbl_raw[:common_Z]
    img_flat = img_flat[:common_Z]; lbl_flat = lbl_flat[:common_Z]

    # Stack channels: (C=2,Z,Y,X) and rechunk to (1,1,H,W) for scrolling
    img = da.stack([img_raw, img_flat], axis=0).rechunk((1, 1, img_raw.shape[-2], img_raw.shape[-1]))
    lbl = da.stack([lbl_raw, lbl_flat], axis=0).rechunk((1, 1, lbl_raw.shape[-2], lbl_raw.shape[-1]))
    annotation_img = da.stack([annotation_img, annotation_img], axis=0).rechunk((1, 1, lbl_raw.shape[-2], lbl_raw.shape[-1]))
    return img, lbl, annotation_img,vp.stem


def rechunk_for_scroll(a):
    # (N,Z,Y,X)  -> (1,1,H,W)
    # (N,C,Z,Y,X)-> (1,1,1,H,W)
    if a.ndim == 4:
        return a.rechunk((1, 1, a.shape[-2], a.shape[-1]))
    elif a.ndim == 5:
        return a.rechunk((1, 1, 1, a.shape[-2], a.shape[-1]))
    return a

def main():
    args = parse_args()


    global OVERWRITE_LABELS
    OVERWRITE_LABELS = args.overwrite_labels
    if args.overwrite_files:
        print("will by default overwrite_labels bc overwrithe_files=True")
        OVERWRITE_LABELS = True

    ALL_VOL_PATHS = sorted(Path(args.vol_dir).glob(args.glob))
    if not ALL_VOL_PATHS:
        raise FileNotFoundError(f"No volumes in {args.vol_dir} matching {args.glob}")

    print(f"[pager] Found {len(ALL_VOL_PATHS)} volumes")


    # ---- Dask runtime knobs (smooth interactivity) ----
    import os, dask
    from dask.cache import Cache
    dask.config.set(scheduler="threads", num_workers=min(6, (os.cpu_count() or 6)))
    DASK_CACHE = Cache(1 * 1024**3)
    DASK_CACHE.register()  # 1 GB cache

    # ---- Initialize viewer ----
    viewer = napari.Viewer(ndisplay=2)

    # Keep handles & index in a tiny mutable namespace
    state = {"idx": 0, "img_layer": None, "lbl_layer": None, "annotation_layer":None,"name": None}

    def _add_current_volume():
        """Load current volume (by state['idx']) and attach to layers (lazy)."""
        from dask import array as da
        vp = ALL_VOL_PATHS[state["idx"]]

        img, lbl, annotation_img, name = load_one_volume(
            vp, z_stride=args.z_stride, overwrite=args.overwrite_files,
            flatten_with=args.flatten_with,
            dir_suffix=args.labels_dir_suffix
        )
        state["name"] = name
        print(f"at current index with vp = {vp}, the shapes of are {[e.shape for e in [img, lbl, annotation_img]]}")

        if img.ndim == 3:         # (Z,Y,X)
            viewer.dims.axis_labels = ("z", "y", "x")
            scale = (1, 1/3, 1)
        else:                     # (C,Z,Y,X)
            viewer.dims.axis_labels = ("channel", "z", "y", "x")
            scale = (1, 1, 1/3, 1)

        # First time: add layers; otherwise, just swap .data
        if state["img_layer"] is None:
            # Axis labels + scale depend on ndim
            # reasonable clims without scanning
            clims = (0, int(np.iinfo(np.uint16).max)) if np.issubdtype(img.dtype, np.integer) else (0.0, 1.0)

            state["img_layer"] = viewer.add_image(
                img, name="OCT_volume",
                colormap="gray", blending="additive",
                rendering="translucent", 
                contrast_limits=clims, scale=scale,
                metadata={"src_path": str(vp)},
            )
            state["lbl_layer"] = viewer.add_labels(
                lbl, name="boundaries",
                opacity=0.5, scale=scale,
            )
            state["lbl_layer"].color = {1:'magenta', 2:'yellow', 3:'cyan', 4:'orange'}

            if annotation_img is not None:
                state["annotation_layer"] = viewer.add_labels( #this might be none
                    annotation_img, name="annotations",
                    opacity=0.5, scale=scale,
                )
            else:
                import pdb; pdb.set_trace()

            # If we have channels, make sure channel axis is a slider (never displayed)
            # ---- Never display the channel axis; keep it as the first slider via dims.order ----
            def lock_channel_axis(viewer: napari.Viewer, chan_axis: int = 0):
                """
                Pin `chan_axis` (e.g., 0 for (C,Z,Y,X)) to the *front* of dims.order
                so it's always a slider and never in the displayed set (which napari
                derives as the last `ndisplay` axes of `order`).
                """
                def _apply(event=None):
                    order = list(viewer.dims.order)
                    if chan_axis in order:
                        if order[0] != chan_axis:
                            order.remove(chan_axis)
                            order.insert(0, chan_axis)
                            viewer.dims.order = tuple(order)
                    else:
                        # Reconstruct a sane order including chan_axis at the front
                        nd = viewer.dims.ndim
                        base = list(range(nd))
                        if chan_axis < nd:
                            base.remove(chan_axis)
                            base.insert(0, chan_axis)
                            viewer.dims.order = tuple(base)
                    # No need to touch viewer.dims.displayed — it's derived automatically

                # Re-enforce after any dims change
                viewer.dims.events.order.connect(_apply)
                viewer.dims.events.ndisplay.connect(_apply)
                _apply()

            # Call this ONLY when you have a channel axis (img.ndim == 4 for (C,Z,Y,X))
            if img.ndim == 4:
                lock_channel_axis(viewer, chan_axis=0)


        else:
            # Swap data in-place (old arrays are GC-able)
            state["img_layer"].data = img
            state["img_layer"].metadata["src_path"] = str(vp)   # <-- add this
            state["lbl_layer"].data = lbl
            if annotation_img is not None:
                # Recall we are sometimes refreshing this info and so we needed to allow a full rewrite.
                if state["annotation_layer"] is None:
                    print("trying to add new annotation_layer")
                    state["annotation_layer"] = viewer.add_labels( #this might be none
                        annotation_img, name="annotations",
                        opacity=0.5, scale=scale,
                    )
                else:
                    state["annotation_layer"].data = annotation_img

        viewer.status = f"[{state['idx']+1}/{len(ALL_VOL_PATHS)}] {state['name']}"

    if args.regenerate_all_volumes_only:
        regenerate_all_volumes(ALL_VOL_PATHS,args)
        print("done regenerating, will now end")
        return

    # Initial load
    _add_current_volume()

    # ---- Keybinds: paginate volumes ----
    # @viewer.bind_key("]")
    def _next_volume(v):
        if state["idx"] + 1 >= len(ALL_VOL_PATHS):
            v.status = "Already at last volume."
            return
        state["idx"] += 1
        _add_current_volume()

    # @viewer.bind_key("[",overwrite=True)
    def _prev_volume(v):
        if state["idx"] == 0:
            v.status = "Already at first volume."
            return
        state["idx"] -= 1
        _add_current_volume()

    # Toggle labels visibility (labels cost a 2nd read per scroll)
    @viewer.bind_key("L")
    def _toggle_labels(v):
        lyr = state["lbl_layer"]
        lyr.visible = not lyr.visible
        v.status = f"Labels visible: {lyr.visible}"

    @viewer.bind_key("Ctrl-D")
    def _drop_current_annotations(v):
        print("Dropping current arrays/layers")
        if state["annotation_layer"] is not None:
            viewer.layers.remove(state["annotation_layer"])
            state["annotation_layer"] = None
        DASK_CACHE.cache.clear()  # <-- this is the one that exists
    # (optional but helpful) also clear napari’s chunk cache
        from napari.components.chunk import chunk_loader
        chunk_loader.cache.clear()
        # state["img_layer"].data = None
        # state["lbl_layer"].data = None

    # use Ctrl+]/Ctrl+[ to paginate
    viewer.bind_key('Ctrl-]', _next_volume, overwrite=True)
    viewer.bind_key('Ctrl-[', _prev_volume, overwrite=True)

    add_segmentation_button(viewer)
    print("now running (pagination: '[' prev, ']' next, 'L' toggle labels)")
    napari.run()

def regenerate_all_volumes(ALL_VOL_PATHS,args):
    """force a regeneration of all volumes, rather than having to wait and click on each one"""
    from dask import array as da
    assert args.overwrite_files == True
    for vp in ALL_VOL_PATHS:
        _, _, _, _ = load_one_volume(
            vp, z_stride=args.z_stride, overwrite=args.overwrite_files,
            flatten_with=args.flatten_with,
            dir_suffix=args.labels_dir_suffix
        )



if __name__ == "__main__":
    main()
