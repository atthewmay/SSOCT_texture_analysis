#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from code_files import file_utils as fu


IMG_KEY = "img"          # npz image key: volume shape (Z, Y, X)
ILM_KEY = "ilm_smooth"   # fixed ILM key in stacked layer npz


def u8(img, lo, hi):
    img = img.astype(np.float32)

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = np.nanpercentile(img, [0.5, 99.5])
        print("fallback lo/hi:", lo, hi)

    x = (img - lo) / max(hi - lo, 1e-6)
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def draw_curve(draw, y, H, W, tile, color, line_px=1):
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(y)
    if ok.sum() < 2:
        return

    x = np.arange(W)[ok]
    y = y[ok]

    pts = [
        (
            int(xi * (tile - 1) / max(W - 1, 1)),
            int(yi * (tile - 1) / max(H - 1, 1)),
        )
        for xi, yi in zip(x, y)
    ]
    draw.line(pts, fill=color, width=line_px)


def render_tile(img, ilm, rpe, lo, hi, tile, line_px):
    H, W = img.shape

    pil = Image.fromarray(u8(img, lo, hi), mode="L")
    pil = pil.resize((tile, tile), Image.Resampling.BILINEAR).convert("RGB")

    draw = ImageDraw.Draw(pil)
    draw_curve(draw, ilm, H, W, tile, color=(0, 255, 0), line_px=line_px)
    draw_curve(draw, rpe, H, W, tile, color=(255, 0, 0), line_px=line_px)

    return pil


def load_panels(image_dir, layers_root, z0, z1, n_panels):
    panels = []

    import re
    seen_ids = set()
    count = 0
    for img_fp in sorted(Path(image_dir).glob("*.img")):
        if count >= n_panels:
            break
        try:
            # vol = np.load(img_fp, mmap_mode="r")[IMG_KEY]
            integer_id = int(re.search(r'(\d+)_Cube', str(img_fp)).group(1))
            if integer_id in seen_ids:
                print(f"skipping {img_fp.name} as the fellow eye has been seen")
                continue
            seen_ids.add(integer_id)
            vol = fu.load_ss_volume2(img_fp)

            layer_fp = fu.new_get_corresponding_layer_path(
                img_fp,
                layers_root=layers_root,
            )
            layers = np.load(layer_fp, mmap_mode="r")

            
            rpe_key = fu.get_algorithm_key_from_filepath(img_fp)

            sample = vol[z0:z1]
            lo, hi = np.nanpercentile(sample, [0.5, 99.5])

            panels.append((img_fp.stem, vol, layers, rpe_key, lo, hi))

            print(f"{img_fp.name}")
            print(f"  layers: {layer_fp.name}")
            print(f"  ILM: {ILM_KEY}")
            print(f"  RPE: {rpe_key}")
            print(f"  vol shape: {vol.shape} dtype: {vol.dtype}")
            print(f"  sample min/max: {np.nanmin(sample)} {np.nanmax(sample)}")
            print(f"  display lo/hi: {lo} {hi}")
            count += 1 
        except:
            print(f"unable to process {img_fp.name}")

    return panels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--layers_root", required=True)
    ap.add_argument("--out", required=True)

    ap.add_argument("--z0", type=int, default=300)
    ap.add_argument("--z1", type=int, default=500)
    ap.add_argument("--z_step", type=int, default=1)

    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--tile", type=int, default=220)
    ap.add_argument("--gap", type=int, default=2)
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--line_px", type=int, default=1)

    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    n_panels = args.cols * args.rows
    panels = load_panels(
        args.image_dir,
        args.layers_root,
        args.z0,
        args.z1,
        n_panels,
    )

    W = args.cols * args.tile + (args.cols - 1) * args.gap
    H = args.rows * args.tile + (args.rows - 1) * args.gap

    with imageio.get_writer(out, mode="I", duration=1 / args.fps, loop=0) as writer:
        for z in range(args.z0, args.z1, args.z_step):
            canvas = Image.new("RGB", (W, H), (0, 0, 0))

            for i, (_, vol, layers, rpe_key, lo, hi) in enumerate(panels):
                r, c = divmod(i, args.cols)

                img = vol[z]
                ilm = layers[ILM_KEY][z]
                rpe = layers[rpe_key][z]

                tile = render_tile(
                    img,
                    ilm,
                    rpe,
                    lo,
                    hi,
                    tile=args.tile,
                    line_px=args.line_px,
                )

                
                if z == args.z0 and i == 0:
                    tile.save(out.with_suffix(".first_tile.png"))
                    print("saved first debug tile:", out.with_suffix(".first_tile.png"))
                    print("first img min/max:", np.nanmin(img), np.nanmax(img))
                    print("first ilm min/max:", np.nanmin(ilm), np.nanmax(ilm))
                    print("first rpe min/max:", np.nanmin(rpe), np.nanmax(rpe))

                x0 = c * (args.tile + args.gap)
                y0 = r * (args.tile + args.gap)
                canvas.paste(tile, (x0, y0))

            draw = ImageDraw.Draw(canvas)
            draw.rectangle((0, H - 24, 80, H), fill=(0, 0, 0))
            draw.text((5, H - 19), f"z={z}", fill=(255, 255, 255))

            writer.append_data(np.asarray(canvas))

            if z % 25 == 0:
                print(f"wrote z={z}")

    print(f"saved {out}")


if __name__ == "__main__":
    main()