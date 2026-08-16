from __future__ import annotations

import re
import argparse
from concurrent.futures import ProcessPoolExecutor
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


def _all_features(saved: dict) -> list[str]:
    features = []
    for group_name in ("extra_maps", "projected_texture_maps", "texture_maps"):
        features.extend(saved.get(group_name, {}).keys())
    return features


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
    ap = argparse.ArgumentParser(description="Save raw OD/OS paired en-face feature maps")
    ap.add_argument("--integer-id", required=True)
    ap.add_argument("--enface-root", required=True)
    ap.add_argument("--features", nargs="+", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tag", default="rawmap")
    ap.add_argument("--dpi", type=int, default=400)
    ap.add_argument("--n-jobs", "--workers", dest="n_jobs", type=int, default=1)
    return ap


def _robust_clim(*imgs, pct=(1, 99)):
    vals = []
    for img in imgs:
        x = np.asarray(img, dtype=float)
        vals.append(x[np.isfinite(x)].ravel())
    vals = np.concatenate([v for v in vals if v.size]) if vals else np.array([])
    if vals.size == 0:
        return None
    lo, hi = np.percentile(vals, pct)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    return float(lo), float(hi)


def _prep_display(arr: np.ndarray, feature_key: str):
    if feature_key == "slab_mean|10->20":
        return np.asarray(arr, dtype=np.float32), "gray"
    img, _, _ = _normalize_overlay(arr)
    return img, "inferno"


def _save_one(arr: np.ndarray, out_path: Path, dpi: int, feature_key: str):
    img, cmap = _prep_display(arr, feature_key)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=dpi)
    ax.imshow(img, cmap=cmap, aspect="equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(out_path, pad_inches=0)
    plt.close(fig)


def _save_pair(
    od_arr: np.ndarray,
    os_arr: np.ndarray,
    out_path: Path,
    dpi: int,
    feature_key: str,
):
    od_img, cmap = _prep_display(od_arr, feature_key)
    os_img, _ = _prep_display(os_arr, feature_key)

    clim = _robust_clim(od_img, os_img)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6), dpi=dpi)
    for a, img, title in zip(ax, (od_img, os_img), ("OD", "OS")):
        im = a.imshow(img, cmap=cmap, aspect="equal")
        if clim is not None:
            im.set_clim(*clim)
        a.set_title(title, fontsize=10)
        a.axis("off")

    fig.suptitle(feature_key, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def _save_pair_job(job):
    feature_key, od_dir, os_dir, outdir, integer_id, tag, dpi = job

    od_saved = fu.load_saved_enface_maps(od_dir)
    os_saved = fu.load_saved_enface_maps(os_dir)

    od_arr = _pick_map(od_saved, feature_key)
    os_arr = _pick_map(os_saved, feature_key)

    out_path = Path(outdir) / f"{integer_id}__OD_OS__{_sanitize(feature_key)}__{tag}.png"
    _save_pair(od_arr, os_arr, out_path, dpi=dpi, feature_key=feature_key)
    return out_path


def main(argv=None):
    args = build_parser().parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    eye_dirs = _find_enface_dirs_by_integer_id(args.enface_root, args.integer_id)

    if "OD" not in eye_dirs or "OS" not in eye_dirs:
        raise FileNotFoundError(f"Need both OD and OS for integer_id={args.integer_id}; found {sorted(eye_dirs)}")

    od_saved = fu.load_saved_enface_maps(eye_dirs["OD"])
    os_saved = fu.load_saved_enface_maps(eye_dirs["OS"])

    features = args.features
    if len(features) == 1 and features[0].lower() == "all":
        features = sorted(set(_all_features(od_saved)) & set(_all_features(os_saved)))

    jobs = [
        (
            feature_key,
            eye_dirs["OD"],
            eye_dirs["OS"],
            outdir,
            args.integer_id,
            args.tag,
            args.dpi,
        )
        for feature_key in features
    ]

    if args.n_jobs == 1:
        out_paths = [_save_pair_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.n_jobs) as ex:
            out_paths = list(ex.map(_save_pair_job, jobs, chunksize=1))

    for p in out_paths:
        print(p)


if __name__ == "__main__":
    main()
