from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

import code_files.ARVO_2026.ArvoFigures2026 as F
from code_files import file_utils as fu

from concurrent.futures import ProcessPoolExecutor, as_completed

SLAB_KEY = "slab_mean|10->20"


def cat_name(x):
    s = re.sub(r"[^A-Z0-9]+", "", str(x).upper())
    if "AIR" in s:
        return "AIR"
    if "AZOOR" in s or "AZOR" in s:
        return "AZOR"
    if "USH2A" in s or "US2A" in s:
        return "RP_USH2A"
    if "RHO" in s:
        return "RP_RHO"
    if "RP" in s:
        return "RP"
    return s or "UNKNOWN"


def load_eye(integer_id, eye, volume_dir, enface_root):
    vp = F._find_volume_by_integer_id_and_eye(volume_dir, integer_id, eye)
    vol = fu.load_ss_volume2(vp, mmap=True)
    saved = fu.load_saved_enface_maps(Path(enface_root) / vp.stem)

    return {
        "full_y": np.mean(vol, axis=1, dtype=np.float32),
        "slab": np.asarray(saved["extra_maps"][SLAB_KEY], dtype=np.float32),
    }


def save_pair(od, os, out_fp, dpi=220, gap_px=8, square_each=False):
    od = np.asarray(od, dtype=np.float32)
    os = np.asarray(os, dtype=np.float32)

    if square_each:
        panel_h = max(od.shape[0], os.shape[0])
        od_panel_w = panel_h
        os_panel_w = panel_h
    else:
        panel_h = max(od.shape[0], os.shape[0])
        od_panel_w = od.shape[1]
        os_panel_w = os.shape[1]

    total_w = od_panel_w + gap_px + os_panel_w

    fig = plt.figure(figsize=(total_w / dpi, panel_h / dpi), dpi=dpi, facecolor="black")

    od_w_frac = od_panel_w / total_w
    gap_w_frac = gap_px / total_w
    os_w_frac = os_panel_w / total_w

    ax1 = fig.add_axes([0, 0, od_w_frac, 1])
    ax2 = fig.add_axes([od_w_frac + gap_w_frac, 0, os_w_frac, 1])

    for ax, im in [(ax1, od), (ax2, os)]:
        ax.imshow(im, cmap="gray",  aspect="auto")
        ax.axis("off")

    out_fp = Path(out_fp)
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fp, pad_inches=0, facecolor="black")
    plt.close(fig)


def process_one(row, volume_dir, enface_root, output_root, dpi, gap_px):
    integer_id = int(row["integer_id"])
    cat = cat_name(row["category"])

    try:
        od = load_eye(integer_id, "OD", volume_dir, enface_root)
        os = load_eye(integer_id, "OS", volume_dir, enface_root)
    except Exception as e:
        return f"skip {integer_id}: {e}"

    stem = f"{cat}__{integer_id}__OD_OS"

    save_pair(
        od["full_y"],
        os["full_y"],
        Path(output_root) / cat / "full_mean_y" / f"{stem}__full_mean_y.png",
        dpi=dpi,
        gap_px=gap_px,
        square_each=True,
    )

    save_pair(
        od["slab"],
        os["slab"],
        Path(output_root) / cat / "slab_mean_10_to_20" / f"{stem}__slab_mean_10_to_20.png",
        dpi=dpi,
        gap_px=gap_px,
        square_each=False,
    )

    return f"saved {integer_id} -> {cat}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--volume_dir", required=True)
    ap.add_argument("--enface_root", required=True)
    ap.add_argument("--output_root", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--gap_px", type=int, default=4)
    args = ap.parse_args()

    df = pd.read_csv(args.metadata)

    if args.split is not None:
        print(f"using split = {args.split}")
        df = df[df["split"].astype(str).str.lower() == args.split.lower()]

    rows = [
        r.to_dict()
        for _, r in df.drop_duplicates("integer_id").iterrows()
    ]

    with ProcessPoolExecutor(max_workers=6) as ex:
        futures = [
            ex.submit(
                process_one,
                r,
                args.volume_dir,
                args.enface_root,
                args.output_root,
                args.dpi,
                args.gap_px,
            )
            for r in rows
        ]

        for fut in as_completed(futures):
            print(fut.result())


if __name__ == "__main__":
    main()