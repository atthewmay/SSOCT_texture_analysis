import argparse
import json
from pathlib import Path

from code_files.texture_package_prod.texture_extraction_utilities import (
    instantiate_fullsize_texture_volumes_from_compact_zarr,
)


def _parse_features_to_keep(text):
    if text is None or str(text).strip() == "":
        return None
    vals = [x.strip() for x in str(text).split(",") if x.strip()]
    return tuple(vals) if vals else None


def _find_manifest_paths(root, manifest_filename="texture_runs_manifest.json"):
    root = Path(root)

    if root.is_file():
        if root.name != manifest_filename:
            raise ValueError(f"If root is a file, it must be {manifest_filename}")
        return [root]

    direct = root / manifest_filename
    if direct.exists():
        return [direct]

    found = sorted(root.rglob(manifest_filename))
    if not found:
        raise FileNotFoundError(f"No {manifest_filename} found under {root}")
    return found


def _canonical_run_paths(
    volume_dir,
    tag,
    texture_filename="texture_bscan_maps.zarr",
    compact_filename="texture_bscan_maps_compact.zarr",
):
    volume_dir = Path(volume_dir)
    run_dir = volume_dir / tag
    compact_zarr_path = run_dir / compact_filename
    zarr_path = run_dir / texture_filename
    return run_dir, compact_zarr_path, zarr_path


def _write_zarr_path_sidecars(volume_dir, manifest_rows, texture_filename="texture_bscan_maps.zarr"):
    volume_dir = Path(volume_dir)

    with open(volume_dir / "zarr_paths.txt", "w") as f:
        for row in manifest_rows:
            tag = row["tag"]
            zarr_path = volume_dir / tag / texture_filename
            f.write(f"{tag}\t{zarr_path}\n")

    if len(manifest_rows) == 1:
        only_tag = manifest_rows[0]["tag"]
        only_path = volume_dir / only_tag / texture_filename
        with open(volume_dir / "zarr_path.txt", "w") as f:
            f.write(str(only_path) + "\n")


def instantiate_one_manifest(
    manifest_path,
    texture_filename="texture_bscan_maps.zarr",
    compact_filename="texture_bscan_maps_compact.zarr",
    features_to_keep=None,
    overwrite=False,
):
    manifest_path = Path(manifest_path)
    volume_dir = manifest_path.parent

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    if not manifest:
        raise ValueError(f"Manifest is empty: {manifest_path}")

    changed = False

    for row in manifest:
        tag = row.get("tag")
        if tag is None:
            raise ValueError(f"Manifest row missing 'tag': {row}")

        run_dir, compact_zarr_path, zarr_path = _canonical_run_paths(
            volume_dir=volume_dir,
            tag=tag,
            texture_filename=texture_filename,
            compact_filename=compact_filename,
        )
        run_dir.mkdir(parents=True, exist_ok=True)

        if compact_zarr_path.exists():
            if zarr_path.exists() and not overwrite:
                print(f"[skip] zarr already exists: {zarr_path}")
            else:
                print(f"[build] {compact_zarr_path} -> {zarr_path}")
                instantiate_fullsize_texture_volumes_from_compact_zarr(
                    compact_zarr_path=compact_zarr_path,
                    out_zarr_path=zarr_path,
                    features_to_keep=features_to_keep,
                    overwrite=overwrite,
                )

            row["compact_zarr_path"] = str(compact_zarr_path)
            row["zarr_path"] = str(zarr_path)
            row["materialized_from_compact_zarr"] = str(compact_zarr_path)
            row["materialized_features_to_keep"] = (
                None if features_to_keep is None else list(features_to_keep)
            )
            changed = True

        elif zarr_path.exists():
            print(f"[reuse] no compact zarr, but zarr already exists: {zarr_path}")
            row["zarr_path"] = str(zarr_path)
            changed = True

        else:
            raise FileNotFoundError(
                f"Neither compact zarr nor zarr found for run '{tag}'.\n"
                f"Looked for:\n"
                f"  compact: {compact_zarr_path}\n"
                f"  zarr:    {zarr_path}"
            )

    if changed:
        backup_path = manifest_path.with_suffix(".json.bak")
        if not backup_path.exists():
            backup_path.write_text(manifest_path.read_text())

        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        _write_zarr_path_sidecars(
            volume_dir,
            manifest,
            texture_filename=texture_filename,
        )
        print(f"[updated] {manifest_path}")

    return manifest_path


def main():
    p = argparse.ArgumentParser(
        description=(
            "Instantiate compact texture sweep outputs into per-run zarr folders "
            "using the local folder structure as the source of truth."
        )
    )
    p.add_argument(
        "root",
        type=str,
        help=(
            "Either:\n"
            "  1) a volume output dir containing texture_runs_manifest.json, or\n"
            "  2) a parent dir under which many such manifests live, or\n"
            "  3) a direct path to texture_runs_manifest.json"
        ),
    )
    p.add_argument(
        "--features-to-keep",
        type=str,
        default=None,
        help="Comma-separated feature names to materialize. Default: all features.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing zarr outputs if they already exist.",
    )
    p.add_argument(
        "--texture-filename",
        type=str,
        default="texture_bscan_maps.zarr",
    )
    p.add_argument(
        "--compact-filename",
        type=str,
        default="texture_bscan_maps_compact.zarr",
    )
    p.add_argument(
        "--manifest-filename",
        type=str,
        default="texture_runs_manifest.json",
    )

    args = p.parse_args()

    features_to_keep = _parse_features_to_keep(args.features_to_keep)
    manifest_paths = _find_manifest_paths(
        args.root,
        manifest_filename=args.manifest_filename,
    )

    print(f"Found {len(manifest_paths)} manifest(s).")
    for mp in manifest_paths:
        print(f"\n=== Processing {mp} ===")
        instantiate_one_manifest(
            manifest_path=mp,
            texture_filename=args.texture_filename,
            compact_filename=args.compact_filename,
            features_to_keep=features_to_keep,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()





