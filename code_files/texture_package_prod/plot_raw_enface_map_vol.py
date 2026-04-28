from __future__ import annotations

import re
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from code_files import file_utils as fu


def _sanitize(s: str) -> str:
    s = re.sub(r"[^\w.-]+", "_", str(s))
    return re.sub(r"_+", "_", s).strip("._-")


def _pick_map(saved: dict, feature_key: str) -> np.ndarray:
    for group_name in ("extra_maps", "projected_texture_maps", "texture_maps"):
        group = saved.get(group_name, {})
        if feature_key in group:
            return np.asarray(group[feature_key], dtype=np.float32)
    raise KeyError(f"{feature_key!r} not found in extra_maps / projected_texture_maps / texture_maps")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Save one raw grayscale en-face feature map as a PNG")
    ap.add_argument("--volume-name", required=True)
    ap.add_argument("--enface-root", required=True)
    ap.add_argument("--feature-key", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tag", default="rawmap")
    ap.add_argument("--dpi", type=int, default=400)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    saved = fu.load_saved_enface_maps(Path(args.enface_root) / args.volume_name)
    arr = _pick_map(saved, args.feature_key)

    out_path = outdir / f"{args.volume_name}__{_sanitize(args.feature_key)}__{args.tag}.png"

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=args.dpi)
    ax.imshow(arr, cmap="gray", aspect="equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(out_path, pad_inches=0)
    plt.close(fig)

    print(out_path)


if __name__ == "__main__":
    main()