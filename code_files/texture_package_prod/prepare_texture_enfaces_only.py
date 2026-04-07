from __future__ import annotations

import argparse
from pathlib import Path

import code_files.file_utils as fu
from code_files.texture_package_prod.texture_enface_utils import prepare_one_volume_texture_enfaces


def parse_slabs(tokens):
    return [tuple(map(int, s.split(":"))) for s in tokens]


def expand_input_tokens(tokens: list[str]) -> list[str]:
    """
    Keep this simple and close to your current workflow:
      - exact file path
      - directory (all *.img inside)
      - glob pattern
    Deduplicate while preserving order.
    """
    out: list[str] = []
    seen: set[str] = set()

    for tok in tokens:
        p = Path(tok)

        if p.is_file():
            s = str(p)
            if s not in seen:
                out.append(s)
                seen.add(s)
            continue

        if p.is_dir():
            matches = sorted(p.glob("*.img"))
            for m in matches:
                s = str(m)
                if s not in seen:
                    out.append(s)
                    seen.add(s)
            continue

        matches = sorted(Path().glob(tok))
        for m in matches:
            if not m.is_file():
                continue
            s = str(m)
            if s not in seen:
                out.append(s)
                seen.add(s)

    if not out:
        raise FileNotFoundError(f"No input volumes found from: {tokens}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--layers_root", required=True)
    ap.add_argument("--enface_out_dir", required=True)

    ap.add_argument("--slabs", nargs="+", default=["10:20"])
    ap.add_argument("--overwrite_flatten", action="store_true")
    ap.add_argument("--include_full_retina", action="store_true")

    ap.add_argument("--texture_window", type=int, default=11)
    ap.add_argument("--texture_step", type=int, default=4)
    ap.add_argument("--texture_levels", type=int, default=16)
    ap.add_argument("--texture_n_jobs", type=int, default=20)

    ap.add_argument("--texture_root_dir", type=str, default=None)
    ap.add_argument("--texture_run", type=str, default=None)
    ap.add_argument("--projected_texture_stat", type=str, default="mean")
    ap.add_argument("--projected_texture_features", type=str, default=None)
    ap.add_argument("--project_texture_n_jobs", type=int, default=20)

    args = ap.parse_args()
    args.slab_offsets = parse_slabs(args.slabs)
    args.inputs = expand_input_tokens(args.inputs)

    enface_out_dir = Path(args.enface_out_dir)
    enface_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(args.inputs)} input volumes")
    for vp in args.inputs:
        print(f"  {vp}")

    vol_results = [
        prepare_one_volume_texture_enfaces(vp, args)
        for vp in args.inputs
    ]

    saved_dirs = []
    for vol_r in vol_results:
        out_dir = fu.save_prepared_enface_maps(vol_r, enface_out_dir)
        saved_dirs.append(out_dir)
        print(f"saved: {out_dir}")

    manifest_path = enface_out_dir / "saved_enface_dirs.txt"
    with open(manifest_path, "w") as f:
        for p in saved_dirs:
            f.write(str(p) + "\n")
    print(f"wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
