# code_files/annotation_tools.py
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from typing import Callable, Dict, Any, Tuple

from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton
from qtpy.QtCore import Qt

def _safe_set_nd(viewer_layer, ndim: int):
    # Shapes/Points in newer napari have .ndim; keep them 3D so they overlay across Z.
    try:
        viewer_layer.ndim = ndim
    except Exception:
        pass
    # Make the shape/point visible across *all* Z if supported
    if hasattr(viewer_layer, "n_dimensional"):
        try:
            viewer_layer.n_dimensional = True
        except Exception:
            pass

def add_annotation_layers(viewer, scale: Tuple[float, ...]):
    """
    Adds:
      - 'ONH_ring' as an empty Shapes layer (ellipse/polygon), visible on all Z
      - 'Fovea'    as a Points layer (single point), visible on all Z

    Returns a dict with layer handles.
    """
    # Ensure 3D (Z,Y,X) to match your image; scale aligns overlays with anisotropy
    onh = viewer.add_shapes(
        name="ONH_ring",
        ndim=3,
        edge_color="yellow",
        face_color=[0, 0, 0, 0],
        edge_width=10,
        opacity=0.9,
        blending="translucent",
        scale=scale,
    )
    _safe_set_nd(onh, 3)

    fov = viewer.add_points(
        name="Fovea",
        ndim=3,
        size=20,
        face_color="red",
        blending="translucent",
        scale=scale,
    )
    _safe_set_nd(fov, 3)

    onh.out_of_slice_display = True
    fov.out_of_slice_display = True

    for lyr in (onh, fov):
        # always visible in 2D slicing no matter which axis is the slider
        if hasattr(lyr, "out_of_slice_display"):
            lyr.out_of_slice_display = True
        # you already call _safe_set_nd(...); this is just a belt-and-suspenders keep-alive
        if hasattr(lyr, "n_dimensional"):
            lyr.n_dimensional = True

    # Convenience keybinds
    @viewer.bind_key("O")
    def _onh_ellipse(_v):
        # Switch to ellipse add mode (you can also select polygon tool in the GUI)
        try:
            onh.mode = "add_ellipse"
        except Exception:
            onh.mode = "add"

    @viewer.bind_key("F")
    def _fovea_add(_v):
        fov.mode = "add"

    return {"onh": onh, "fovea": fov}


def clear_annotations(viewer):
    """Empty both layers if present."""
    for name in ("ONH_ring", "Fovea"):
        if name in viewer.layers:
            lyr = viewer.layers[name]
            if hasattr(lyr, "data"):
                try:
                    lyr.data = [] if hasattr(lyr.data, "__len__") else None
                except Exception:
                    pass


def _layer_xy_from_img(img_layer):
    """
    Return a tuple (arr, has_channel) where arr is (Z,Y,X) dask/numpy array
    for en-face slab, and whether the first axis is a channel to strip.
    """
    arr = img_layer.data
    if arr.ndim == 3:
        # (Z,Y,X)
        return arr, False
    elif arr.ndim == 4:
        # (C,Z,Y,X) -> use channel 0 (raw) for en-face
        return arr[0], True
    else:
        raise ValueError(f"Unexpected image ndim={arr.ndim}; expected 3 or 4")


def add_enface_slab_updater(viewer, img_layer, slab: int = 9, name: str = "enface_slab"):
    """
    Adds a thin-slab MIP image that updates when Z slider moves.
    Set slab<=0 to skip.
    """
    if slab <= 0:
        return None

    arr3d, _ = _layer_xy_from_img(img_layer)
    half = max(1, slab // 2)

    # Initialize with current z
    def _compute_enface(zc: int):
        import numpy as np
        try:
            import dask.array as da
            is_dask = isinstance(arr3d, da.Array)
        except Exception:
            is_dask = False

        z0 = max(0, zc - half)
        z1 = min(arr3d.shape[0], zc + half + 1)
        slab_arr = arr3d[z0:z1]  # (z1-z0, Y, X)
        if is_dask:
            return slab_arr.max(axis=0)  # stays lazy; napari will compute as needed
        else:
            return np.max(slab_arr, axis=0)

    # Find current z
    z_axis = None
    labels = list(getattr(viewer.dims, "axis_labels", []))
    if labels:
        z_axis = labels.index("z") if "z" in labels else 0
    curr_z = viewer.dims.current_step[z_axis] if z_axis is not None else 0

    enface = viewer.add_image(
        _compute_enface(curr_z),
        name=name,
        colormap="gray",
        blending="additive",
        opacity=0.8,
        scale=img_layer.scale[-2:] if img_layer.scale is not None else (1, 1),
        visible=True,
    )

    # Update when z changes
    def _on_step(event=None):
        steps = viewer.dims.current_step
        z = steps[z_axis] if z_axis is not None else 0
        enface.data = _compute_enface(z)

    viewer.dims.events.current_step.connect(_on_step)
    return enface


def add_annotation_save_dock(
    viewer,
    get_context: Callable[[], Dict[str, Any]],
    out_dir: Path,
):
    """
    Adds a small dock widget with:
      - Save (writes JSON with fovea + ONH ring)
      - Clear (resets both layers)
    Also registers Ctrl+S to save.
    """

    out_dir.mkdir(parents=True, exist_ok=True)

    def _collect_payload(meta: Dict[str, Any]) -> Dict[str, Any]:
        payload: Dict[str, Any] = dict(meta)

        # Fovea (0 or 1 point). Napari stores points in data with shape (N, D)
        fov_yxz = None
        if "Fovea" in viewer.layers:
            pts = viewer.layers["Fovea"].data
            if pts is not None and len(pts) > 0:
                # Keep the last point if multiple were placed
                fyxz = np.asarray(pts[-1]).tolist()  # [Z,Y,X]
                fov_yxz = fyxz
        payload["fovea"] = {"zyx": fov_yxz}

        # ONH ring: support ellipse or polygon; store vertex list in YX (2D) + note type
        ring = None
        ring_type = None
        if "ONH_ring" in viewer.layers:
            sh = viewer.layers["ONH_ring"]
            # Take the first shape if many; or you could save all
            if len(sh.data) > 0:
                ring = np.asarray(sh.data[0])[:, -2:].tolist()  # drop Z if present; keep (Y,X)
                try:
                    ring_type = sh.shape_type[0]
                except Exception:
                    ring_type = "polygon"

        payload["onh_ring"] = {"vertices_yx": ring, "type": ring_type}
        return payload

    def _save():
        meta = get_context()
        payload = _collect_payload(meta)
        name = meta.get("name", "unknown")
        path = (out_dir / f"{name}.ann.json")
        with path.open("w") as f:
            json.dump(payload, f, indent=2)
        viewer.status = f"Saved annotations → {path}"

    def _clear():
        clear_annotations(viewer)
        viewer.status = "Cleared ONH_ring and Fovea layers."

    # Dock UI
    w = QWidget()
    layout = QVBoxLayout(w); layout.setAlignment(Qt.AlignTop)
    bsave = QPushButton("Save annotation (Ctrl+S)")
    bclear = QPushButton("Clear annotation")
    layout.addWidget(bsave); layout.addWidget(bclear)
    bsave.clicked.connect(_save)
    bclear.clicked.connect(_clear)
    viewer.window.add_dock_widget(w, area="right", name="Annotations")

    @viewer.bind_key("Ctrl-S", overwrite=True)
    def _save_hotkey(_v):
        _save()

    return w


def load_annotations_into_layers(viewer, ann_dir: Path, vol_name: str, onh_layer, fov_layer) -> bool:
    """
    Load saved annotations for `vol_name` from `ann_dir` (the JSON created by the Save button)
    and insert them into the provided ONH (Shapes) and Fovea (Points) layers.

    Returns True if anything was loaded.
    """
    import json
    import numpy as np

    ann_path = ann_dir / f"{vol_name}.ann.json"
    if not ann_path.exists():
        return False

    with ann_path.open("r") as f:
        payload = json.load(f)

    # Pick a Z to attach the ring vertices; prefer the saved z_current, else current Z
    z0 = payload.get("z_current")
    try:
        labels = list(getattr(viewer.dims, "axis_labels", []))
        z_axis = labels.index("z") if "z" in labels else 0
    except Exception:
        z_axis = 0
    if z0 is None:
        z0 = int(viewer.dims.current_step[z_axis])

    loaded_any = False

    # ---- ONH ring ----
    try:
        ring = payload.get("onh_ring", {})
        verts_yx = ring.get("vertices_yx")
        if verts_yx:
            v = np.asarray(verts_yx, dtype=float)
            v_zyx = np.column_stack([np.full(len(v), float(z0)), v])  # (N,3) as Z,Y,X
            # Replace any existing geometry (keep it simple)
            onh_layer.data = [v_zyx]   # load as polygon (robust across napari versions)
            loaded_any = True
    except Exception as e:
        viewer.status = f"Error loading ONH: {e}"

    # ---- Fovea point ----
    try:
        fov = payload.get("fovea", {})
        zyx = fov.get("zyx")
        if zyx is not None:
            arr = np.asarray(zyx, dtype=float).reshape(1, 3)
            fov_layer.data = arr
            loaded_any = True
    except Exception as e:
        viewer.status = f"Error loading Fovea: {e}"

    # Seed pinner bases so slice-pinning uses these as canonical geometry
    if loaded_any:
        try:
            onh_layer.metadata["base_data"] = [np.array(s, dtype=float, copy=True) for s in onh_layer.data]
            fov_layer.metadata["base_data"] = np.array(fov_layer.data, dtype=float, copy=True)
        except Exception:
            pass

    return loaded_any


def build_3d_annotation_masks(
    viewer,
    img_layer,
    onh_layer,
    fov_layer,
    fovea_diam: int = 10,
    name: str = "AnnotationMask",
):
    """
    Create/update a (Z,Y,X) labels volume from current ONH ellipse + Fovea point.
      - ONH: polygon in (Z,X) → extruded across all Y → label 1 (yellow)
      - Fovea: disk in (Z,X) with given diameter → extruded across Y → label 2 (red)

    The layer is rebuilt each time you call this (cheap; dask-lazy).
    """
    import numpy as np

    try:
        from skimage.draw import polygon2mask, disk
    except Exception as e:
        raise RuntimeError(
            "scikit-image is required for mask rasterization. `pip install scikit-image`"
        ) from e

    # Figure out (Z,Y,X) shape + (Z,Y,X) scale from the image layer
    data = img_layer.data
    if data.ndim == 3:                      # (Z,Y,X)
        Z, H, W = data.shape
        _sc = getattr(img_layer, "scale", None)
        scale = tuple(_sc) if _sc is not None else (1, 1, 1)
        print("processing withe 3 dims")
    elif data.ndim == 4:                    # (C,Z,Y,X)
        Z, H, W = data.shape[1], data.shape[-2], data.shape[-1]
        _sc = getattr(img_layer, "scale", None) or (1, 1, 1, 1)
        scale = (_sc[1], _sc[2], _sc[3])
    else:
        raise ValueError(f"Unexpected image ndim={data.ndim}; expected 3 or 4.")

    # ---- Build 2D masks in (Z,X) ----
    # onh_zx = None
    # if onh_layer is not None and len(onh_layer.data) > 0:
    #     v = np.asarray(onh_layer.data[0], dtype=float)  # (N,3) as (Z,Y,X) vertices
    #     if v.shape[1] >= 3:
    #         v_zx = v[:, [0, 2]]                         # drop Y → (Z,X)
    #         onh_zx = polygon2mask((Z, W), v_zx).astype(np.uint8)  # (Z,W), 1 inside
    #     # else: leave as None

    import termplotlib as tpl
    onh_zx = None
    if onh_layer is not None and len(onh_layer.data) > 0:
        ring_zx_meta = onh_layer.metadata.get("ring_zx", None)
        if ring_zx_meta is not None:
            print("ring_zx_meta is not None:")
            v_zx = np.asarray(ring_zx_meta, dtype=float)              # (N,2): Z,X
        else:
            v_full = np.asarray(onh_layer.data[0], dtype=float)       # (N,3)
            print(f"and onh_layer.data={onh_layer.data}")
            print(f"v_full is {v_full} with shape {v_full.shape}")
            # If this came from legacy YX save, Z will be constant → degenerate; mask will be empty
            v_zx = v_full[:, [0, 2]]

        vals, counts = np.unique(v_zx, return_counts=True)
        print(v_zx)
        print(vals,counts)
        fig = tpl.figure()
        fig.barh(counts, vals, force_ascii=True)
        fig.show()
        # Clip to bounds to be safe
        v_zx[:, 0] = np.clip(v_zx[:, 0], 0, Z - 1)
        v_zx[:, 1] = np.clip(v_zx[:, 1], 0, W - 1)
        print(v_zx)
        onh_zx = polygon2mask((Z, W), v_zx).astype(np.uint8)          # (Z,W)
        print(onh_zx.shape)
        print(onh_zx)
        # compute the dask array


    fov_zx = None
    if fov_layer is not None and fov_layer.data is not None and len(fov_layer.data) > 0:
        p = np.asarray(fov_layer.data[-1], dtype=float)  # last point (Z,Y,X)
        cz, cx = int(round(p[0])), int(round(p[2]))
        rr, cc = disk((cz, cx), max(1, int(round(fovea_diam / 2))), shape=(Z, W))
        fov_zx = np.zeros((Z, W), dtype=np.uint8)
        fov_zx[rr, cc] = 1

    # ---- Extrude across Y to get (Z,Y,X) ----
    # try:
    import dask.array as da
    zeros3 = da.zeros((Z, H, W), dtype=np.uint8, chunks=(1, H, W))
    if onh_zx is not None:
        onh3 = da.repeat(da.from_array(onh_zx, chunks=(1, W))[:, None, :], H, axis=1)  # label 1
    else:
        onh3 = zeros3
        print("set onh to zeros")
    if fov_zx is not None:
        fov3 = da.repeat(da.from_array(fov_zx, chunks=(1, W))[:, None, :], H, axis=1)  # label 2
    else:
        fov3 = zeros3
        print("set fov to zeros")
    # Combine: fovea overrides ONH where overlapping
    combined = da.maximum(onh3, (fov3 * 2))

    # ---- Add/update Labels layer ----
    if name in viewer.layers:
        print("expecting the label already exists")
        lyr = viewer.layers[name]
        lyr.data = combined
        lyr.visible = True
    else:
        print("adding the labels")
        lyr = viewer.add_labels(
            combined, name=name, opacity=0.5, blending="translucent", scale=scale, visible=True
        )
        lyr.color = {0: (0, 0, 0, 0), 1: "yellow", 2: "red"}

        @viewer.bind_key("M", overwrite=True)
        def _toggle_masks(v):
            lyr.visible = not lyr.visible
            v.status = f"{name} visible: {lyr.visible}"

    return lyr
