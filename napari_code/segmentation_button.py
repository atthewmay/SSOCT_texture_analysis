
from __future__ import annotations
import os
import time
import tempfile
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import napari
from qtpy.QtWidgets import QWidget, QPushButton, QVBoxLayout, QMessageBox

import code_files.file_utils as fu

# --- CONFIG ---
# Absolute path to the *external* Python interpreter (other conda env)
EXTERNAL_PY = "/Users/matthewhunt/miniconda3/envs/han_air/bin/python"


# Path to the external runner script (relative to THIS file)
RUNNER_SCRIPT = (Path(__file__).resolve().parents[1] / "code_files" / "segmentation_code" / "seg_runner_new.py")

# Context-free EZ runner. This does not run a segmentation pipeline or build ctx.
EZ_RUNNER_SCRIPT = (Path(__file__).resolve().parents[1] / "code_files" / "segmentation_code" / "ez_runner_napari.py")

# Where the final multipage PDF should land by default
DEFAULT_REPORT_DIR = (Path(__file__).resolve().parents[1] / "reports")  # repo_root/reports
DEFAULT_EZ_REPORT_DIR = DEFAULT_REPORT_DIR / "ez_finder"

# Used by the EZ finder to load the actual layer NPZ, not napari-painted label overlays.
# Prefer metadata['layers_root'] from napari_run_with_layers_pagination.py when available.
DEFAULT_LAYERS_ROOT = None
for _k in ("layers_root"):
    if hasattr(fu, "C") and _k in fu.C:
        DEFAULT_LAYERS_ROOT = Path(fu.C[_k])
        print(f"Setting default layers root as {DEFAULT_LAYERS_ROOT}")
        break

# Optional: directory of mini volumes for the external runner (can override in UI call)
DEFAULT_MINI_ROOT = (Path(__file__).resolve().parents[1] / "data_all_volumes_extra_mini")  # <<< EDIT ME


def _active_image_layer(viewer: napari.Viewer):
    layer = viewer.layers.selection.active
    if layer is None:
        print("hey, the layer is none, check this out")
    if layer is None or not isinstance(layer, napari.layers.Image):
        # fall back to first Image layer
        for L in viewer.layers:
            if isinstance(L, napari.layers.Image):
                return L
        return None
    return layer


def _current_z_index(viewer: napari.Viewer, layer) -> Optional[int]:
    """Return the current slice index for a 3D (Z,Y,X) image layer.
    Note: napari Layer objects don't expose `.viewer`; use the passed-in viewer.
    """
    data = layer.data
    if getattr(data, 'ndim', 0) == 4:
        z = int(viewer.dims.current_step[1])
    elif getattr(data, 'ndim', 0) == 3:
        z = int(viewer.dims.current_step[0])
    elif getattr(data, 'ndim', 0) != 3:
        return None
    # Assume first axis is Z for (Z, Y, X) data
    print(f"using z slice at {z}")
    return z


def _save_current_slice(layer, z_index: int) -> Path:
    """Realize the currently shown 2D slice to a temp .npy, returning its path."""
    if layer.data.ndim == 4: # Ignore the leading flattened channel
        data_to_use = layer.data[0,:,:,:]
    else:
        data_to_use = layer.data
    arr2d = np.asarray(data_to_use[z_index])  # triggers dask compute for one slice if dask-backed
    arr2d = np.ascontiguousarray(arr2d)
    tmpdir = Path(tempfile.gettempdir())
    tname = f"napari_slice_{int(time.time())}_{os.getpid()}.npy"
    out = tmpdir / tname
    np.save(out, arr2d)
    return out






def _save_line_to_temp(yline: np.ndarray, prefix: str = "napari_layer_line") -> Path:
    """Save a 1D layer line to temp .npy."""
    yline = np.asarray(yline, dtype=np.float32).squeeze()
    if yline.ndim != 1:
        raise ValueError(f"Expected 1D layer line, got shape {yline.shape}")

    tmpdir = Path(tempfile.gettempdir())
    tname = f"{prefix}_{int(time.time())}_{os.getpid()}.npy"
    out = tmpdir / tname
    np.save(out, yline)
    return out


def _resolve_layers_root_from_image_layer(layer) -> Path:
    """Find the layers_root needed to load the original layer NPZ.

    Preferred source is image-layer metadata set by napari_run_with_layers_pagination.py:
        metadata["layers_root"] = str(args.layers_root)

    Fallback is code_files/CONSTANTS.yaml if it has a usable layer-root key.
    """
    md = getattr(layer, "metadata", {}) or {}
    for key in ("layers_root", "layer_root", "layers_dir"):
        val = md.get(key, None)
        if val:
            return Path(val)

    if DEFAULT_LAYERS_ROOT is not None:
        return Path(DEFAULT_LAYERS_ROOT)

    raise ValueError(
        "Could not resolve layers_root. Add metadata['layers_root'] in "
        "napari_run_with_layers_pagination.py when adding the OCT image layer, "
        "or add layers_root to CONSTANTS.yaml."
    )


def _selected_rpe_layer_name_from_img_path(img_path: Path) -> str:
    """Use the same ID-to-algorithm lookup used elsewhere in the project."""
    return fu.get_algorithm_key_from_filepath(img_path)


def _load_layer_line_from_npz(
    *,
    img_path: Path,
    image_layer,
    z_index: int,
    z_stride: int,
    layer_name: str,
) -> tuple[np.ndarray, Path, int]:
    """Load the true 1D layer curve from the original *_stacked.npz layer file.

    This intentionally does NOT read the napari Labels overlay. The label overlay
    is a painted integer volume derived from the curves; here we go back to the
    source layer NPZ used by ensure_nonflat_artifacts / ensure_labels_zarr.
    """
    layers_root = _resolve_layers_root_from_image_layer(image_layer)
    layer_path = fu.new_get_corresponding_layer_path(img_path, layers_root=layers_root)

    if not layer_path.exists():
        raise FileNotFoundError(f"Layer NPZ not found for {img_path}: {layer_path}")

    source_z = int(z_index) * int(z_stride)
    with np.load(layer_path) as layers:
        if layer_name not in layers.files:
            raise KeyError(
                f"Layer {layer_name!r} not found in {layer_path}. "
                f"Available: {list(layers.files)}"
            )

        arr = np.asarray(layers[layer_name], dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Expected layer {layer_name!r} to have shape (Z,W), got {arr.shape}")
        # if source_z < 0 or source_z >= arr.shape[0]:
        #     raise IndexError(
        #         f"Requested source_z={source_z}, but {layer_name!r} has Z={arr.shape[0]}. "
        #         f"napari z_index={z_index}, z_stride={z_stride}"
        #     )

        try:
            yline = arr[source_z].astype(np.float32, copy=False)
        except IndexError:
            print(
                f"Requested source_z={source_z}, but {layer_name!r} has Z={arr.shape[0]}. "
                f"napari z_index={z_index}, z_stride={z_stride}"
            )
            print("We'll now try just using the abs z index")
            yline = arr[z_index].astype(np.float32,copy = False)

    return yline, layer_path, source_z


def _save_rpe_line_from_layer_npz(
    *,
    img_path: Path,
    image_layer,
    z_index: int,
    z_stride: int,
    rpe_layer_name: str | None = None,
) -> tuple[Path, str, Path, int]:
    """Resolve final RPE layer via ID2algo lookup and save true NPZ curve to temp."""
    layer_name = rpe_layer_name or _selected_rpe_layer_name_from_img_path(img_path)
    if "stargardt" in str(img_path):
        print("\n\n\t\t\tdetected we are trialing on the newer stargardt test data, reverting to just using 'original'")
        layer_name = 'original_method_y2_vertical_shifted'
    yline, layer_path, source_z = _load_layer_line_from_npz(
        img_path=img_path,
        image_layer=image_layer,
        z_index=z_index,
        z_stride=z_stride,
        layer_name=layer_name,
    )
    return _save_line_to_temp(yline, prefix="napari_rpe_line"), layer_name, layer_path, source_z


def _try_save_ilm_line_from_layer_npz(
    *,
    img_path: Path,
    image_layer,
    z_index: int,
    z_stride: int,
) -> Path | None:
    """Best-effort true ILM curve load from layer NPZ.

    EZ finder can run without ILM; if ILM is missing, just return None.
    """
    for layer_name in ["ilm_smooth"]:
        try:
            yline, _, _ = _load_layer_line_from_npz(
                img_path=img_path,
                image_layer=image_layer,
                z_index=z_index,
                z_stride=z_stride,
                layer_name=layer_name,
            )
            return _save_line_to_temp(yline, prefix=f"napari_{layer_name}")
        except Exception as e:
            print(f"[EZFinderButton] Could not load optional {layer_name}: {e}")
    return None


def _build_subprocess_cmd(
    current_bscan_npy: Path,
    out_pdf: Path,
    img_src_path,
    z_index,
    mini_root: Optional[Path] = None,
    k_samples: int = 5,
    downsample: float = 1.5,
    max_workers: int = 8,
    debug: bool = False,   # <-- ADD
    run_extra_slices: bool = False,
    z_stride: int = 1,
    RPE_OR_ILM: str = "RPE",
    annotation_root: Optional[Path] = None,
) -> list[str]:
    mini_root = Path(mini_root) if mini_root is not None else DEFAULT_MINI_ROOT
    cmd = [
        str(EXTERNAL_PY), str(RUNNER_SCRIPT),
        "--current-bscan", str(current_bscan_npy),
        "--out", str(out_pdf),
        "--mini-root", str(mini_root),
        "--k", str(k_samples),
        "--downsample", str(downsample),
        "--max-workers", str(max_workers),
        "--img_src_path",str(img_src_path),
        "--z_index",str(z_index),
        "--z_stride",str(z_stride),
        "--RPE_OR_ILM",str(RPE_OR_ILM),
    ]

    if annotation_root is not None:
        cmd += ["--annotation_root", str(annotation_root)]

    if debug:
        # Run under pdb and force debug mode in the script
        cmd = [cmd[0]] +  ["-m", "pdb", "-c", "continue", '--'] + cmd[1:]
        cmd += [ "--debug"]
    if run_extra_slices:
        cmd += [ "--run_extra_slices"]

    return cmd


import traceback

def safe_breakpoint():
    # Prints the stack then drops you into pdb from the console thread
    try:
        raise RuntimeError("debug breakpoint")
    except RuntimeError:
        traceback.print_exc()
    import pdb; pdb.set_trace()

# inside your callback instead of pdb.set_trace():

class SegmentationButton(QWidget):
    """Small QWidget with one button to launch the external runner."""
    def __init__(self, viewer: napari.Viewer, mini_root: Optional[Path] = None,
                 debug_mode = False,
                run_extra_slices = False,
                 title = "Run ILM+RPE PDF on Current Slice",
                 z_stride=1,
                 RPE_OR_ILM="RPE",
                 annotation_root: Optional[Path] = None,
                 ):  # flip to True when you want pdb
        super().__init__()
        self.viewer = viewer
        self.mini_root = Path(mini_root) if mini_root else DEFAULT_MINI_ROOT
        self.debug_mode = debug_mode
        self.run_extra_slices = run_extra_slices

        self.button = QPushButton(title)
        self.button.clicked.connect(self._on_click)

        layout = QVBoxLayout()
        layout.addWidget(self.button)
        self.setLayout(layout)
        self.z_stride = z_stride
        self.RPE_OR_ILM = RPE_OR_ILM 
        self.annotation_root = Path(annotation_root) if annotation_root is not None else None

    def _on_click(self):
        try:
            layer = _active_image_layer(self.viewer)
            if layer is None:
                QMessageBox.warning(self, "No image layer", "Select an image layer.")
                return
            if getattr(layer.data, 'ndim', 0) not in [3,4]:
                QMessageBox.warning(self, "Not 3D/4D", f"Expected a 3D or 4D ([ch],Z,Y,X) image, and yours is {layer.data.shape}")
                return

            # z = _current_z_index(layer)
            z = _current_z_index(self.viewer, layer)

            if z is None:
                QMessageBox.warning(self, "No Z index", "Could not determine current slice.")
                return


            # from IPython import embed
            # embed(colors="neutral")
            if "src_path" in layer.metadata:
                img_path = Path(layer.metadata["src_path"])

            # Fallback: if layer was loaded via a napari reader plugin, it may have .source.path
            elif getattr(layer, "source", None) and getattr(layer.source, "path", None):
                img_path = Path(layer.source.path)
                print("unable to get the src_path directly from the Path(layer.metadata")
            else:
                QMessageBox.warning(self, "Missing source path",
                                    "This layer has no recorded file path (metadata['src_path']).")
                # return
            # Realize and save the current slice to a temp .npy
            npy_path = _save_current_slice(layer, z)

            # Make output path
            DEFAULT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            out_pdf = DEFAULT_REPORT_DIR / 'seg_reports' / f"seg_report_{timestamp}.pdf"
            print(f"will save output pdf at {out_pdf}")

            # Build and launch external process

            # Build and launch external process
            cmd = _build_subprocess_cmd(
                current_bscan_npy=npy_path,
                out_pdf=out_pdf,
                mini_root=self.mini_root,
                k_samples=5,
                downsample=1.5,
                max_workers=8,
                debug=self.debug_mode,
                run_extra_slices = self.run_extra_slices,
                img_src_path=img_path,
                z_index = z,
                z_stride = self.z_stride,
                RPE_OR_ILM=self.RPE_OR_ILM,
                annotation_root=self.annotation_root,

            )

            if self.debug_mode:
                # Foreground: napari UI will block; use the terminal you launched napari from
                subprocess.run(cmd, check=False)
            else:
                # Non-blocking normal run
                subprocess.Popen(cmd)
                self.viewer.status = f"Launched external segmentation job → {out_pdf.name}"


            # # Non-blocking launch; we don't wait or capture stdout (no UI output desired)
            # subprocess.Popen(cmd)
            # self.viewer.status = f"Launched external segmentation job → {out_pdf.name}"
        except Exception as e:
            # QMessageBox.critical(self, "Launch error", str(e))
            tb = traceback.format_exc()
            print(tb)  # goes to terminal if you launched napari from Terminal
            QMessageBox.critical(self, "Launch error", tb)
            # (optional) re-raise during development so napari debug tools catch it
            # raise




class EZFinderButton(QWidget):
    """Run the context-free EZ finder on the active B-scan and loaded RPE layer.

    This intentionally bypasses seg_runner_new and all step_/ctx machinery.
    It pulls:
      - the current B-scan from the active image layer
      - the final RPE curve directly from the original layer NPZ

    The selected RPE layer name is resolved from file_utils.get_algorithm_key_from_filepath,
    so it follows the ID-to-algorithm lookup. It intentionally does not read
    the napari painted Labels overlay.
    """

    def __init__(
        self,
        viewer: napari.Viewer,
        *,
        debug_mode: bool = False,
        title: str = "Run EZ Finder on Current Slice",
        z_stride: int = 1,
        rpe_layer_name: str | None = None,
        rpe_label_value: int | None = None,
        n_jobs: int = 6,
    ):
        super().__init__()
        self.viewer = viewer
        self.debug_mode = debug_mode
        self.z_stride = z_stride
        self.rpe_layer_name = rpe_layer_name
        self.rpe_label_value = rpe_label_value
        self.n_jobs = n_jobs

        self.button = QPushButton(title)
        self.button.clicked.connect(self._on_click)

        layout = QVBoxLayout()
        layout.addWidget(self.button)
        self.setLayout(layout)

    def _build_ez_cmd(self, bscan_npy: Path, rpe_line_npy: Path, out_npz: Path, out_png: Path, ilm_line_npy: Path | None = None) -> list[str]:
        """Build command for ez_runner_napari.py.

        Parameters are intentionally hard-coded/swept inside ez_runner_napari.py.
        This button only passes the current B-scan, selected RPE line, and output paths.
        """
        cmd = [
            str(EXTERNAL_PY), str(EZ_RUNNER_SCRIPT),
            "--current-bscan", str(bscan_npy),
            "--rpe-line", str(rpe_line_npy),
            "--out-npz", str(out_npz),
            "--out-png", str(out_png),
            "--n-jobs", str(self.n_jobs),
        ]
        if ilm_line_npy is not None:
            cmd += ["--ilm-line", str(ilm_line_npy)]
        if self.debug_mode:
            cmd = [cmd[0]] + ["-m", "pdb", "-c", "continue", "--"] + cmd[1:]
        return cmd

    def _on_click(self):
        try:
            layer = _active_image_layer(self.viewer)
            if layer is None:
                QMessageBox.warning(self, "No image layer", "Select an image layer.")
                return
            if getattr(layer.data, "ndim", 0) not in [3, 4]:
                QMessageBox.warning(self, "Not 3D/4D", f"Expected 3D/4D image, got {layer.data.shape}")
                return

            z = _current_z_index(self.viewer, layer)
            if z is None:
                QMessageBox.warning(self, "No Z index", "Could not determine current slice.")
                return

            if "src_path" in layer.metadata:
                img_path = Path(layer.metadata["src_path"])
            elif getattr(layer, "source", None) and getattr(layer.source, "path", None):
                img_path = Path(layer.source.path)
            else:
                QMessageBox.warning(self, "Missing source path", "Image layer has no metadata['src_path'].")
                return

            rpe_line_npy, rpe_layer_name, layer_path, source_z = _save_rpe_line_from_layer_npz(
                img_path=img_path,
                image_layer=layer,
                z_index=z,
                z_stride=self.z_stride,
                rpe_layer_name=self.rpe_layer_name,
            )
            ilm_line_npy = _try_save_ilm_line_from_layer_npz(
                img_path=img_path,
                image_layer=layer,
                z_index=z,
                z_stride=self.z_stride,
            )
            bscan_npy = _save_current_slice(layer, z)

            print(
                f"[EZFinderButton] using true NPZ layer {rpe_layer_name!r} "
                f"from {layer_path} at source_z={source_z}"
            )

            DEFAULT_EZ_REPORT_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            out_npz = DEFAULT_EZ_REPORT_DIR / f"ez_result_{timestamp}.npz"
            out_png = DEFAULT_EZ_REPORT_DIR / f"ez_result_{timestamp}.png"

            cmd = self._build_ez_cmd(bscan_npy, rpe_line_npy, out_npz, out_png, ilm_line_npy=ilm_line_npy)
            print("[EZFinderButton] running:", " ".join(map(str, cmd)))

            if self.debug_mode:
                # Foreground: napari UI will block; use the terminal you launched napari from.
                subprocess.run(cmd, check=False)
            else:
                # Non-blocking normal run. Outputs are tagged files plus a manifest CSV/JSON.
                subprocess.Popen(cmd)
                manifest_csv = out_npz.with_name(f"{out_npz.stem}__manifest.csv")
                self.viewer.status = f"Launched EZ parameter sweep → {manifest_csv.name}"

        except Exception:
            tb = traceback.format_exc()
            print(tb)
            QMessageBox.critical(self, "EZ finder error", tb)


def add_ez_finder_button(viewer: napari.Viewer, z_stride=1):
    """Factory function for the context-free EZ finder button."""
    widget = EZFinderButton(
        viewer,
        debug_mode=False,
        title="Run EZ Finder Sweep on Current Slice",
        z_stride=z_stride,
        n_jobs=6,
    )
    viewer.window.add_dock_widget(widget, area="right", name="EZ Finder Sweep")

    widget_dbg = EZFinderButton(
        viewer,
        debug_mode=True,
        title="Run EZ Finder Sweep DEBUG",
        z_stride=z_stride,
        n_jobs=1,
    )
    viewer.window.add_dock_widget(widget_dbg, area="right", name="EZ Finder Sweep DEBUG")
    return widget


def add_segmentation_button(viewer: napari.Viewer,annotation_root=None,z_stride=1) -> SegmentationButton:
    """Factory function you can call from your napari plugin to add the button."""
    mini_root="/Users/matthewhunt/Research/Iowa_Research/Han_AIR/data_all_volumes/"
    widget = SegmentationButton(viewer,mini_root=mini_root,debug_mode=False,title="Run RPE Layers on Current Slice",z_stride=z_stride,annotation_root=annotation_root)
    viewer.window.add_dock_widget(widget, area='right', name='ILM+RPE PDF')

    widget = SegmentationButton(viewer,mini_root=mini_root,debug_mode=False,title="Run ILM Layers on Current Slice",z_stride=z_stride,RPE_OR_ILM="ILM",annotation_root=annotation_root)
    viewer.window.add_dock_widget(widget, area='right', name='ILM PDF')


    widget = SegmentationButton(viewer,mini_root=mini_root,debug_mode=True,title="Run RPE Layers on Current Slice DEBUG",z_stride=z_stride,annotation_root=annotation_root)
    viewer.window.add_dock_widget(widget, area='right', name='RPE PDF DEBUG')

    widget = SegmentationButton(viewer,mini_root=mini_root,debug_mode=True,title="Run ILM Layers on Current Slice DEBUG",z_stride=z_stride,RPE_OR_ILM="ILM",annotation_root=annotation_root)
    viewer.window.add_dock_widget(widget, area='right', name='ILM PDF DEBUG')

    widget = SegmentationButton(viewer,mini_root=mini_root,debug_mode=False,title="Run Layers on Current+Extra Slices",run_extra_slices=True,z_stride=z_stride,annotation_root=annotation_root)
    viewer.window.add_dock_widget(widget, area='right', name='ILM+RPE PDF Extra Slices')

    widget = SegmentationButton(viewer,mini_root=mini_root,debug_mode=True,title="Run Layers on Current+Extra Slices DEBUG",run_extra_slices=True,z_stride=z_stride,annotation_root=annotation_root)
    viewer.window.add_dock_widget(widget, area='right', name='ILM+RPE PDF Extra Slices DEBUG')

    add_ez_finder_button(viewer,z_stride=z_stride)
    return widget