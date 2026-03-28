import zarr
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))  # adds Han_AIR/ to path
import napari
import numpy as np

#!/usr/bin/env python3
import argparse
import code_files.file_utils as fu

from segmentation_button import add_segmentation_button

from code_files.zarr_file_utils import ensure_nonflat_artifacts, ensure_flattened_artifacts
# Multithreading hack for older python
import sys, importlib.util
main = sys.modules['__main__']
if not hasattr(main, '__spec__'):
    main.__spec__ = importlib.util.spec_from_loader('__main__', loader=None)

C = fu.load_constants()

def parse_args():
    p = argparse.ArgumentParser(
        description="Load an OCT volume and its corresponding layer predictions"
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
        # default="*.npy",
        help="Glob pattern inside --vol_dir to find volumes (default: *.npy)."
    )

    # p.add_argument("--cube_numbers", type=int, nargs="+", default=[], help="List of ints")
    p.add_argument("--cube_numbers", type=str, default=None)

    p.add_argument(
        "--layers_root",
        type=str,
        help="supply entire dir"
    )

    p.add_argument(
        "--annotation_root",
        type=str,
        default=None,
    )

    p.add_argument(
        "--flattened_artifacts_root",
        type=Path,
        default=Path("/Volumes/T9/iowa_research/Han_AIR_Dec_2025/flattened_artifacts"),
    )



    p.add_argument(
        "--view_mode",
        type=str,
        choices=["nonflat", "flat"],
        default="nonflat",
        help="Choose whether to load native or flattened artifacts.",
    )


    p.add_argument('--overwrite_files',
            action="store_true",
            help="overwrite all zaars"
    )

    p.add_argument('--overwrite_labels',
            action="store_true",
            help="overwrite just the label zaars"
    )

    p.add_argument(
            "--z_stride",
            type=int,
            default=1,
            help="Thin Z by this stride when writing Zarr (keeps XY native)."
    )
    p.add_argument(
        "--use_skip_yaml",
        action="store_true",
    )





    return p.parse_args()

### ==================Slice-Labelling Utils==========
# ---- QUICK GRAB HOTKEYS (writes YAML immediately; safe/atomic) ----
    
GRAB_YAML_PATH = Path("/Users/matthewhunt/Research/Iowa_Research/Han_AIR/data/example_slice_yamls/choroidal_EZ_good_slices.yaml")
# GRAB_YAML_PATH.parent.mkdir(parents=True, exist_ok=True)
print(f"[quick-grab] YAML will be written to: {GRAB_YAML_PATH}")

def _atomic_write_yaml(data: dict, path: Path):
    import yaml
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    tmp.replace(path)  # atomic on same filesystem

def _load_grabs(path: Path) -> dict:
    import yaml
    if not path.exists():
        return {}
    try:
        with open(path, "r") as f:
            d = yaml.safe_load(f) or {}
        return d if isinstance(d, dict) else {}
    except Exception as e:
        print(f"[quick-grab] WARNING: failed to read {path}: {e}")
        return {}

def _current_z_index(viewer: napari.Viewer, img_ndim: int) -> int:
    # img.ndim == 3: (Z,Y,X) -> z axis is 0
    # img.ndim == 4: (C,Z,Y,X) -> z axis is 1
    z_axis = 0 if img_ndim == 3 else 1
    return int(viewer.dims.current_step[z_axis])

import time
from dask import array as da
def from_zarr_fresh(path: str):
    za = zarr.open_array(path, mode="r", cache_metadata=False)
    # include mtime so unchanged files still benefit from caching
    mtime = Path(path).stat().st_mtime_ns
    return da.from_zarr(za, name=f"from_zarr-{mtime}-{time.time_ns()}")

def load_one_volume(
    vp: Path,
    *,
    view_mode: str,
    z_stride: int,
    overwrite: bool,
    layers_root: str | Path,
    annotation_root: str | Path | None = None,
    flattened_artifacts_root: Path = Path('/Volumes/T9/iowa_research/Han_AIR_Dec_2025/flattened_artifacts'),
):
    """
    Return (img, lbl, annotation_img, name) for one volume.

    nonflat mode:
        img : dask array (Z,Y,X)
        lbl : dict[str, dask array] from native label zarr-group

    flat mode:
        img : dask array (Z,Y,X)
        lbl : dask array (Z,Y,X) flattened painted labels
    """
    from dask import array as da

    vp = Path(vp)

    if view_mode not in {"nonflat", "flat"}:
        raise ValueError(f"unknown view_mode: {view_mode}")


    flattener_name = None
    if view_mode == "nonflat":
        artifacts = ensure_nonflat_artifacts(
            vp,
            layers_root=layers_root,
            annotation_root=annotation_root,
            z_stride=1,              # keep cached raw artifacts full-Z
            overwrite=overwrite,
            make_image_zarr=True,
            make_label_zarr=True,
            make_annotation_zarr=True,
        )

        img = da.from_zarr(str(artifacts["image_zarr"]))
        img = img[::z_stride]
        img = img.rechunk((1, img.shape[-2], img.shape[-1]))

        g = zarr.open_group(str(artifacts["label_zarr"]), mode="r")
        lbl = {
            name: da.from_zarr(g[name])[::z_stride].rechunk((1, g[name].shape[-2], g[name].shape[-1]))
            for name in g.array_keys()
        }

    else:
        flattener_name = fu.get_algorithm_key_from_filepath(vp)
        artifacts = ensure_flattened_artifacts(
            vp,
            flatten_with=flattener_name,
            layers_root=layers_root,
            annotation_root=annotation_root,
            flattened_artifacts_root=flattened_artifacts_root,
            z_stride=z_stride,
            overwrite=overwrite,
            make_image_zarr=True,
            make_label_zarr=True,
            save_flat_layers_npz=True,
            make_annotation_zarr=True,
        )

        img = da.from_zarr(str(artifacts["image_zarr"])+'/data')
        img = img.rechunk((1, img.shape[-2], img.shape[-1]))

        lbl = da.from_zarr(str(artifacts["label_zarr"])+'/data')
        lbl = lbl.rechunk((1, lbl.shape[-2], lbl.shape[-1]))

    annotation_img = None
    if artifacts["annotation_zarr"] is not None:
        annotation_img = from_zarr_fresh(str(artifacts["annotation_zarr"]))
        annotation_img = annotation_img[::z_stride]
        annotation_img = annotation_img.rechunk(
            (1, annotation_img.shape[-2], annotation_img.shape[-1])
        )

    return img, lbl, annotation_img, vp.stem,flattener_name


def main():
    args = parse_args()
    


    global OVERWRITE_LABELS # We no longer use this actually
    OVERWRITE_LABELS = args.overwrite_labels
    if args.overwrite_files:
        print("will by default overwrite_labels bc overwrithe_files=True")
        OVERWRITE_LABELS = True

    ALL_VOL_PATHS = fu.get_all_vol_paths(args.vol_dir,glob=args.glob,cube_numbers=args.cube_numbers,use_skip_yaml=args.use_skip_yaml)

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
    state = {
        "idx": 0,
        "img_layer": None,
        "lbl_layers": {},
        "flat_lbl_layer": None,
        "annotation_layer": None,
        "name": None,
    }


    def _append_grab(category: str):
        z_step_size = int(getattr(args, "z_stride", 1))
        vol_id = str(state.get("name", "UNKNOWN_VOLUME"))

        # Determine current z from the currently loaded volume layer
        img_layer = state.get("img_layer", None)
        if img_layer is None:
            print("[quick-grab] No image layer yet; nothing saved.")
            return
        img_ndim = img_layer.data.ndim
        slice_idx = _current_z_index(viewer, img_ndim)

        d = _load_grabs(GRAB_YAML_PATH)
        d.setdefault(category, {})
        d[category].setdefault(vol_id, [])
        d[category][vol_id].append([slice_idx, z_step_size])

        _atomic_write_yaml(d, GRAB_YAML_PATH)
        print(f"[quick-grab] + {category} | {vol_id}: [{slice_idx}, {z_step_size}]  ->  {GRAB_YAML_PATH}")



    def _add_current_volume():
        """Load current volume (by state['idx']) and attach to layers (lazy)."""
        from dask import array as da
        vp = ALL_VOL_PATHS[state["idx"]]

        img, lbl, annotation_img, name,flattener_name = load_one_volume(
                vp,
                view_mode=args.view_mode,
                z_stride=args.z_stride,
                overwrite=args.overwrite_files,
                layers_root=args.layers_root,
                annotation_root=args.annotation_root,
                flattened_artifacts_root=args.flattened_artifacts_root,
            )


        state["name"] = name

        viewer.dims.axis_labels = ("z", "y", "x")
        scale = (1, 1/3, 1)
        viewer.dims.order = (0, 1, 2)
        cur = list(viewer.dims.current_step)
        cur[0] = img.shape[0] // 2
        viewer.dims.current_step = tuple(cur)

        # First time: add layers; otherwise, just swap .data
        
        if state["img_layer"] is None:
            clims = (0, int(np.iinfo(np.uint16).max)) if np.issubdtype(img.dtype, np.integer) else (0.0, 1.0)

            state["img_layer"] = viewer.add_image(
                img,
                name="OCT_volume",
                colormap="gray",
                blending="additive",
                rendering="translucent",
                contrast_limits=clims,
                scale=scale,
                metadata={"src_path": str(vp)},
            )

            if args.view_mode == "nonflat":
                state["lbl_layers"] = {}
                for set_name, arr in lbl.items():
                    lyr = viewer.add_labels(arr, name=set_name, opacity=0.8, scale=scale)
                    lyr.color = {0:(0,0,0,0), 1:'magenta', 2:'green', 6:'yellow', 3:'cyan', 4:'orange', 5:'blue'}
                    lyr.color_mode = "direct"
                    lyr.refresh()
                    state["lbl_layers"][set_name] = lyr
            else:
                state["flat_lbl_layer"] = viewer.add_labels(
                    lbl,
                    name=f"{flattener_name}_flat_labels",
                    opacity=0.8,
                    scale=scale,
                )
                state["flat_lbl_layer"].color = {0:(0,0,0,0), 1:'magenta', 2:'green', 6:'yellow', 3:'cyan', 4:'orange', 5:'blue'}
                state["flat_lbl_layer"].color_mode = "direct"
                state["flat_lbl_layer"].refresh()

            if annotation_img is not None:
                state["annotation_layer"] = viewer.add_labels(
                    annotation_img,
                    name="annotations",
                    opacity=0.5,
                    scale=scale,
                )

        else:
            state["img_layer"].data = img
            state["img_layer"].metadata["src_path"] = str(vp)

            if args.view_mode == "nonflat":
                if state["flat_lbl_layer"] is not None:
                    viewer.layers.remove(state["flat_lbl_layer"])
                    state["flat_lbl_layer"] = None

                cur_keys = set(state["lbl_layers"].keys())
                new_keys = set(lbl.keys())
                if cur_keys != new_keys:
                    for lyr in list(state["lbl_layers"].values()):
                        viewer.layers.remove(lyr)
                    state["lbl_layers"] = {}
                    for set_name, arr in lbl.items():
                        lyr = viewer.add_labels(arr, name=set_name, opacity=0.8, scale=scale)
                        lyr.color = {0:(0,0,0,0), 1:'magenta', 2:'green', 6:'yellow', 3:'cyan', 4:'orange', 5:'blue'}
                        lyr.color_mode = "direct"
                        lyr.refresh()
                        state["lbl_layers"][set_name] = lyr
                else:
                    for set_name, arr in lbl.items():
                        state["lbl_layers"][set_name].data = arr

            else:
                for lyr in list(state["lbl_layers"].values()):
                    viewer.layers.remove(lyr)
                state["lbl_layers"] = {}

                if state["flat_lbl_layer"] is None:
                    state["flat_lbl_layer"] = viewer.add_labels(
                        lbl,
                        name=f"{flattener_name}_flat_labels",
                        opacity=0.8,
                        scale=scale,
                    )
                    state["flat_lbl_layer"].color = {0:(0,0,0,0), 1:'magenta', 2:'green', 6:'yellow', 3:'cyan', 4:'orange', 5:'blue'}
                    state["flat_lbl_layer"].color_mode = "direct"
                    state["flat_lbl_layer"].refresh()
                else:
                    state["flat_lbl_layer"].data = lbl

            if annotation_img is not None:
                if state["annotation_layer"] is None:
                    state["annotation_layer"] = viewer.add_labels(
                        annotation_img,
                        name="annotations",
                        opacity=0.5,
                        scale=scale,
                    )
                else:
                    state["annotation_layer"].data = annotation_img

        print("order:", viewer.dims.order, "displayed:", viewer.dims.displayed, "labels:", viewer.dims.axis_labels)
        viewer.status = f"[{state['idx']+1}/{len(ALL_VOL_PATHS)}] {state['name']}"

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

    @viewer.bind_key("Ctrl-H", overwrite=True)
    def _grab_choroid(v):
        _append_grab("choroidal_grab")

    @viewer.bind_key("Ctrl-E", overwrite=True)
    def _grab_ez(v):
        _append_grab("EZ_grab")

    @viewer.bind_key("Ctrl-G", overwrite=True)
    def _grab_good(v):
        _append_grab("good_seg")

    @viewer.bind_key("Ctrl-S", overwrite=True)
    def _grab_hypersmooth_fail(v):
        _append_grab("hypersmooth_fail")



    # use Ctrl+]/Ctrl+[ to paginate
    viewer.bind_key('Ctrl-]', _next_volume, overwrite=True)
    viewer.bind_key('Ctrl-[', _prev_volume, overwrite=True)

    add_segmentation_button(viewer,z_stride=args.z_stride)
    print("now running (pagination: '[' prev, ']' next, 'L' toggle labels)")
    napari.run()



if __name__ == "__main__":
    main()
