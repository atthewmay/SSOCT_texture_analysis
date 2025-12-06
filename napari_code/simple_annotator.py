#!/usr/bin/env python3
"""
Minimal en‑face ONH+Fovea annotator (tiny, opinionated, easy to hack)

Assumptions (edit these few constants if your data differs):
- Input files: .npy arrays shaped (Z, X, Y).
- En‑face image: mean across Y (axis = Y_AXIS).
- One ONH polygon + one Fovea point per volume.
- Output: uint8 labels (Z, X, Y) saved as .zarr (chunked).
- Labels: 0=bg, 1=ONH, 2=Fovea (fovea is a small disk extruded through Y).

Keys: S = save & next,  N = next (no save),  Q = quit

Usage:
python simple_annotator.py --glob "~/data/vols/*.npy" --outdir "~/data/labels" --limit 0
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

import napari
from skimage.draw import polygon2mask, disk
import zarr

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))  # adds Han_AIR/ to path
import code_files.file_utils as fu

# ----------------------- tweakables (single place) ----------------------- #
Y_AXIS = 1              # index of Y in (Z, X, Y)
FOVEA_RADIUS = 8        # pixels in (Z, X)
CHUNKS = (64, 64, 64)   # zarr chunking for (Z, X, Y)
LABEL_ONH = 1
LABEL_FOVEA = 2

# ----------------------------- tiny helpers ----------------------------- #

def load_volume_npy(path: Path) -> np.ndarray:
    arr = np.load(path,allow_pickle=True)
    assert arr.ndim == 3, f"expected 3-D array, got {arr.shape} from {path}"
    return arr


def enface_mean(vol: np.ndarray) -> np.ndarray:
    img = vol.mean(axis=Y_AXIS).astype(np.float32)
    m, M = float(img.min()), float(img.max())
    return (img - m) / (M - m + 1e-8)


def rasterize_labels_2d(zx_shape, poly_rc, fovea_rc) -> np.ndarray:
    Z, X = zx_shape
    lab = np.zeros((Z, X), dtype=np.uint8)
    if poly_rc is not None:
        coords = np.asarray(poly_rc)
        if len(coords) == 4:  
            from skimage.draw import polygon2mask, ellipse
            # napari ellipse: bounding box with 4 corner coords
            rmin, cmin = coords.min(axis=0)
            rmax, cmax = coords.max(axis=0)
            r_center = (rmin + rmax) / 2
            c_center = (cmin + cmax) / 2
            r_radius = (rmax - rmin) / 2
            c_radius = (cmax - cmin) / 2
            rr, cc = ellipse(r_center, c_center, r_radius, c_radius, shape=(Z, X))
            lab[rr, cc] = LABEL_ONH
        else:
            # assume polygon
            mask = polygon2mask((Z, X), coords)
            lab[mask] = LABEL_ONH
    


    # if poly_rc is not None and len(poly_rc) >= 3:
    #     print(f"polygon2mask((Z, X), np.asarray(poly_rc)) is {polygon2mask((Z, X), np.asarray(poly_rc))}")
    #     print(f"uniques={np.unique(polygon2mask((Z, X), np.asarray(poly_rc)),return_counts=True)}")
    #     lab[polygon2mask((Z, X), np.asarray(poly_rc))] = LABEL_ONH
    if fovea_rc is not None and len(fovea_rc) == 2:
        r, c = map(int, np.round(fovea_rc))
        rr, cc = disk((r, c), FOVEA_RADIUS, shape=(Z, X))
        lab[rr, cc] = np.maximum(lab[rr, cc], LABEL_FOVEA)
    return lab


def extrude_y(labels2d: np.ndarray, y_len: int) -> np.ndarray:
    return np.repeat(labels2d[:, None, :], y_len, axis=1)


def save_zarr(path: Path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        import shutil; shutil.rmtree(path, ignore_errors=True)
    z = zarr.open(path, mode='w', shape=arr.shape, chunks=CHUNKS, dtype='uint8')
    z[:] = arr

# ------------------------------- napari UI ------------------------------- #
class Session:
    def __init__(self, files, outdir: Path):
        self.files = files
        self.outdir = outdir
        self.i = 0
        self.viewer = None
        self.img_layer = None
        self.onh_layer = None
        self.fov_layer = None
        self.cur_shape = None  # (Z, X, Y)

    def start(self):
        self.viewer = napari.Viewer()
        self.viewer.bind_key('ctrl+s')(self._save)
        self.viewer.bind_key('ctrl+n')(self._next)
        self.viewer.bind_key('ctrl+b')(self._back)
        self.viewer.bind_key('ctrl+q')(self._quit)
        self._load()
        napari.run()

    def _load(self):
        if self.i >= len(self.files):
            print('Done.')
            self.viewer.close(); return
        p = self.files[self.i]
        print(f"loading file at: {p}")
        vol = fu.load_ss_volume2(p)
        self.cur_shape = vol.shape  # (Z, X, Y)
        img = enface_mean(vol)      # (Z, X), float32 in [0,1]

        title = f"{p.name} — en‑face(mean Y)"
        if self.img_layer is None:
            self.img_layer = self.viewer.add_image(img, name=title)
        else:
            self.img_layer.data = img
            self.img_layer.name = title

        # reset annotation layers
        if self.onh_layer is not None: self.viewer.layers.remove(self.onh_layer)
        if self.fov_layer is not None: self.viewer.layers.remove(self.fov_layer)
        self.onh_layer = self.viewer.add_shapes(name='ONH', shape_type='polygon', edge_width=2)
        self.fov_layer = self.viewer.add_points(name='Fovea', size=8)
        self.viewer.status = f"[{self.i+1}/{len(self.files)}] {p} — S=save,next  N=next  Q=quit"

    def _save_current(self):
        Z,  Y, X = self.cur_shape
        poly = self.onh_layer.data[0] if len(self.onh_layer.data) else None
        fpt  = self.fov_layer.data[0] if len(self.fov_layer.data) else None
        lab2 = rasterize_labels_2d((Z, X), poly, fpt)
        # import matplotlib as plt
        # plt.figure()
        # plt.imshow(lab2)
        # plt.show()
        lab3 = extrude_y(lab2, Y).astype(np.uint8)

        stem = Path(self.files[self.i]).stem
        out = (self.outdir / stem).with_suffix('.labels.zarr')
        save_zarr(out, lab3)
        meta = {
            'source': str(self.files[self.i]),
            'save_path': str(out),
            'shape': list(lab3.shape),
            'y_axis': Y_AXIS,
            'fovea_radius': FOVEA_RADIUS,
            'labels': {'ONH': LABEL_ONH, 'FOVEA': LABEL_FOVEA},
        }
        with open((self.outdir / stem).with_suffix('.labels.json'), 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"saved → {out}  shape={lab3.shape}")

    # key bindings
    def _save(self, _=None):
        print("attempting save")
        self._save_current(); 
        # self.i += 1; 
        # self._load()
    def _next(self, _=None):
        self.i += 1; self._load()
        
    def _back(self, _=None):
        self.i -= 1; self._load()

    def _quit(self, _=None):
        self.viewer.close()

# --------------------------------- CLI --------------------------------- #

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description='Minimal en‑face annotator')
    ap.add_argument('--glob', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--limit', type=int, default=0)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    files = [Path(p).expanduser() for p in sorted(__import__('glob').glob(str(Path(args.glob).expanduser())))]
    if args.limit: files = files[:args.limit]
    if not files: raise SystemExit('no files matched --glob')
    outdir = Path(args.outdir).expanduser(); outdir.mkdir(parents=True, exist_ok=True)
    Session(files, outdir).start()

if __name__ == '__main__':
    main()