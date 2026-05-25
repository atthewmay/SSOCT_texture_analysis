from __future__ import annotations

import re
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from code_files import file_utils as fu
from code_files.ARVO_2026.ArvoFigures2026 import _normalize_overlay


def _sanitize(s: str) -> str:
    s = re.sub(r"[^\w.-]+", "_", str(s))
    return re.sub(r"_+", "_", s).strip("._-")


def _pick_map(saved: dict, feature_key: str) -> np.ndarray:
    for group_name in ("extra_maps", "projected_texture_maps", "texture_maps"):
        group = saved.get(group_name, {})
        if feature_key in group:
            return np.asarray(group[feature_key], dtype=np.float32)
    raise KeyError(f"{feature_key!r} not found in extra_maps / projected_texture_maps / texture_maps")


def _find_enface_dirs_by_integer_id(enface_root: str | Path, integer_id: str | int) -> dict[str, Path]:
    enface_root = Path(enface_root)
    integer_id = str(integer_id)

    out = {}
    for p in sorted(enface_root.iterdir()):
        if not p.is_dir():
            continue
        try:
            if fu.get_integer_id(p.name) == integer_id:
                eye = fu.get_eye(p.name).upper()
                out[eye] = p
        except Exception:
            continue

    if not out:
        raise FileNotFoundError(f"No enface dirs found for integer_id={integer_id} under {enface_root}")
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Save raw grayscale en-face feature maps for both eyes")
    ap.add_argument("--integer-id", required=True)
    ap.add_argument("--enface-root", required=True)
    # ap.add_argument("--feature-key", required=True)
    ap.add_argument("--features", nargs="+", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tag", default="rawmap")
    ap.add_argument("--dpi", type=int, default=400)
    return ap


def _save_one(arr: np.ndarray, out_path: Path, dpi: int,feature_key:str):
    if feature_key == "slab_mean|10->20":
        cmap = 'gray'
        img = arr
    else:
        cmap = 'inferno'
        img,_,_ = _normalize_overlay(arr)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=dpi)
    ax.imshow(img, cmap=cmap, aspect="equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(out_path, pad_inches=0)
    plt.close(fig)


def main(argv=None):
    args = build_parser().parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    eye_dirs = _find_enface_dirs_by_integer_id(args.enface_root, args.integer_id)

    for eye in ("OD", "OS"):
        for feature_key in args.features:
            if eye not in eye_dirs:
                continue

            saved = fu.load_saved_enface_maps(eye_dirs[eye])
            arr = _pick_map(saved, feature_key)

            out_path = outdir / f"{args.integer_id}_{eye}__{_sanitize(feature_key)}__{args.tag}.png"
            _save_one(arr, out_path, dpi=args.dpi,feature_key=feature_key)
            print(out_path)


if __name__ == "__main__":
    main()