from __future__ import annotations

import argparse
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle
from scipy import ndimage
from skimage.transform import AffineTransform, warp

from code_files.texture_package_prod.texture_enface_utils import make_enface_isotropic_x


# ---------------------------------------------------------------------
# Repo imports
# ---------------------------------------------------------------------

def _find_repo_root(start: Path) -> Path:
    for cand in [start, *start.parents]:
        if (cand / "code_files").exists():
            return cand
    return start


REPO_ROOT = _find_repo_root(Path(__file__).resolve())
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

try:
    from code_files.texture_package_prod.texture_regions import (
        make_etdrs_grid_plus_rings,
        summarize_by_regions,
    )
except Exception as e:
    raise ImportError(
        "Could not import texture_regions.py from the repo. "
        "Run this script from inside the repo or edit REPO_ROOT/sys.path."
    ) from e


# ---------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------

def parse_eye_from_name(name: str) -> str:
    m = re.search(r"(?:^|[_-])(OD|OS)(?:[_.-]|$)", name.upper())
    if not m:
        raise ValueError(f"Could not find OD or OS in filename: {name}")
    return m.group(1)


def canonical_case_id(stem: str) -> str:
    for suf in ("_enface_maps", "__enface_maps", "_enface", "_maps"):
        if stem.endswith(suf):
            return stem[: -len(suf)]
    return stem


def resolve_annotation_path(case_stem: str, annotation_root: Path) -> Path:
    cand = [
        annotation_root / f"{case_stem}.labels.zarr",
        annotation_root / f"{case_stem}.zarr",
    ]
    for p in cand:
        if p.exists():
            return p
    raise FileNotFoundError(f"No annotation zarr found for {case_stem} under {annotation_root}")


def load_enface_maps(path: Path) -> dict[str, np.ndarray]:
    """
    Supports either:
      1) single .npz with keys -> 2D maps
      2) directory of .npy maps
    """
    if path.is_file() and path.suffix == ".npz":
        with np.load(path) as obj:
            return {k: np.asarray(obj[k], dtype=np.float32) for k in obj.files}

    if path.is_dir():
        npys = sorted(path.glob("*.npy"))
        if not npys:
            raise FileNotFoundError(f"No .npy feature maps in {path}")
        return {p.stem: np.load(p).astype(np.float32) for p in npys}

    raise ValueError(f"Unsupported enface input: {path}")


def discover_cases(enface_root: Path, glob_pat: str) -> list[tuple[str, Path]]:
    cases: list[tuple[str, Path]] = []

    for p in sorted(enface_root.glob(glob_pat)):
        if p.is_file() and p.suffix == ".npz":
            cases.append((canonical_case_id(p.stem), p))
    if cases:
        return cases

    for d in sorted(enface_root.iterdir()):
        if not d.is_dir():
            continue
        npzs = sorted(d.glob(glob_pat))
        npzs = [p for p in npzs if p.is_file() and p.suffix == ".npz"]
        if npzs:
            if len(npzs) != 1:
                print(f"[{d.name}] expected 1 npz, found {len(npzs)}; using first")
            cases.append((canonical_case_id(d.name), npzs[0]))

    if not cases:
        raise FileNotFoundError(f"No cases found under {enface_root}")
    return cases


# ---------------------------------------------------------------------
# Annotation / mask helpers
# ---------------------------------------------------------------------

def load_annotation_enface(annotation_path: Path) -> np.ndarray:
    import dask.array as da

    z = da.from_zarr(str(annotation_path))
    proj = z.max(axis=1).compute()
    return np.asarray(proj)


def centroid_from_mask(mask: np.ndarray) -> tuple[float, float]:
    if mask.sum() == 0:
        raise ValueError("Mask is empty; centroid undefined")
    rr, cc = ndimage.center_of_mass(mask)
    return float(cc), float(rr)


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    rr, cc = np.where(mask)
    if rr.size == 0:
        raise ValueError("Mask is empty; bbox undefined")
    return int(rr.min()), int(rr.max()), int(cc.min()), int(cc.max())


def build_onh_and_lateral_exclusion(
    onh_mask: np.ndarray,
    fovea_xy: tuple[float, float],
) -> np.ndarray:
    out = onh_mask.astype(bool).copy()
    r0, r1, c0, c1 = bbox_from_mask(onh_mask)
    onh_cx = 0.5 * (c0 + c1)
    fovea_x = float(fovea_xy[0])
    strip = np.zeros_like(out, dtype=bool)

    if onh_cx >= fovea_x:
        strip[r0 : r1 + 1, c1:] = True
    else:
        strip[r0 : r1 + 1, : c0 + 1] = True

    out |= strip
    return out


# ---------------------------------------------------------------------
# Rigid geometry helpers
# ---------------------------------------------------------------------

def flip_lr_image_and_points(
    arr: np.ndarray,
    points_xy: dict[str, tuple[float, float]],
) -> tuple[np.ndarray, dict[str, tuple[float, float]]]:
    out = np.fliplr(arr)
    w = arr.shape[1]
    pts = {k: (float(w - 1 - x), float(y)) for k, (x, y) in points_xy.items()}
    return out, pts


def build_rigid_standardize_transform(
    input_shape: tuple[int, int],
    fovea_xy: tuple[float, float],
    onh_xy: tuple[float, float],
    output_shape: tuple[int, int] | None = None,
) -> tuple[AffineTransform, dict[str, tuple[float, float]], float]:
    h, w = input_shape
    if output_shape is None:
        output_shape = (h, w)

    oh, ow = output_shape
    fx, fy = map(float, fovea_xy)
    ox, oy = map(float, onh_xy)
    theta = np.arctan2(oy - fy, ox - fx)
    dtheta = -theta

    t1 = AffineTransform(translation=(-fx, -fy))
    r = AffineTransform(rotation=dtheta)
    t2 = AffineTransform(translation=((ow - 1) / 2.0, (oh - 1) / 2.0))
    A = AffineTransform(matrix=(t2.params @ r.params @ t1.params))

    pts = {}
    for name, (x, y) in {"fovea": fovea_xy, "onh": onh_xy}.items():
        xp, yp = A(np.array([[x, y]], dtype=np.float32))[0]
        pts[name] = (float(xp), float(yp))

    return A, pts, float(np.degrees(dtheta))


def warp_image(
    arr: np.ndarray,
    A: AffineTransform,
    output_shape: tuple[int, int],
    order: int = 1,
    cval: float = np.nan,
) -> np.ndarray:
    out = warp(
        arr.astype(np.float32, copy=False),
        inverse_map=A.inverse,
        output_shape=output_shape,
        order=order,
        preserve_range=True,
        cval=cval,
    )
    return out.astype(np.float32, copy=False)


def warp_mask(mask: np.ndarray, A: AffineTransform, output_shape: tuple[int, int]) -> np.ndarray:
    out = warp(
        mask.astype(np.float32),
        inverse_map=A.inverse,
        output_shape=output_shape,
        order=0,
        preserve_range=True,
        cval=0,
    )
    return out > 0.5


# ---------------------------------------------------------------------
# Region helpers
# ---------------------------------------------------------------------

def make_sector_masks(shape: tuple[int, int], fovea_xy: tuple[float, float]) -> dict[str, np.ndarray]:
    cx, cy = map(float, fovea_xy)
    rr, cc = np.indices(shape)
    dr = rr - cy
    dc = cc - cx
    theta = (np.degrees(np.arctan2(-dr, dc)) + 360.0) % 360.0
    return {
        "superior": (theta >= 45) & (theta < 135),
        "nasal": (theta < 45) | (theta >= 315),
        "inferior": (theta >= 225) & (theta < 315),
        "temporal": (theta >= 135) & (theta < 225),
    }


def add_sectorized_ring_masks(
    masks: dict[str, np.ndarray],
    fovea_xy: tuple[float, float],
) -> dict[str, np.ndarray]:
    out = dict(masks)
    sectors = make_sector_masks(next(iter(masks.values())).shape, fovea_xy)
    ring_like = [
        k for k in masks
        if k == "center" or k.endswith("_ring") or k.startswith("extra_ring_")
    ]

    for ring_name in ring_like:
        if ring_name == "center":
            continue
        for sec_name, sec_mask in sectors.items():
            out[f"{ring_name}_{sec_name}"] = masks[ring_name] & sec_mask

    return out


def ordered_region_names(masks: dict[str, np.ndarray]) -> list[str]:
    out = ["center"]

    for ring_name in ("inner", "outer"):
        for sec in ("temporal", "superior", "nasal", "inferior"):
            key = f"{ring_name}_{sec}"
            if key in masks:
                out.append(key)

    extra_ids = sorted(
        int(k.split("_")[2])
        for k in masks
        if k.startswith("extra_ring_") and k.count("_") == 2
    )
    for i in extra_ids:
        for sec in ("temporal", "superior", "nasal", "inferior"):
            key = f"extra_ring_{i}_{sec}"
            if key in masks:
                out.append(key)

    for sec in ("temporal", "superior", "nasal", "inferior"):
        key = f"outer_region_{sec}"
        if key in masks:
            out.append(key)

    if "whole" in masks:
        out.append("whole")

    return out


def build_region_value_map(
    masks: dict[str, np.ndarray],
    region_means: dict[str, float],
) -> np.ndarray:
    shape = next(iter(masks.values())).shape
    out = np.full(shape, np.nan, dtype=np.float32)

    for name in ordered_region_names(masks):
        if name == "whole":
            continue
        mask = masks.get(name)
        if mask is None:
            continue
        val = region_means.get(name, np.nan)
        if np.isfinite(val):
            out[mask] = float(val)

    return out


def apply_exclusion_to_masks(
    masks: dict[str, np.ndarray],
    exclusion_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    return {k: (v & ~exclusion_mask) for k, v in masks.items()}


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def _normalize_for_display(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    good = np.isfinite(x)
    if not good.any():
        return np.zeros_like(x)
    lo, hi = np.nanpercentile(x[good], [2, 98])
    if hi <= lo:
        hi = lo + 1e-6
    x = np.clip((x - lo) / (hi - lo), 0, 1)
    x[~good] = 0
    return x


def draw_etdrs_overlay(
    ax,
    image: np.ndarray,
    fovea_xy: tuple[float, float],
    radii: tuple[float, float, float],
    extra_radii: tuple[float, ...],
    exclusion_mask: np.ndarray | None = None,
    title: str | None = None,
):
    ax.imshow(_normalize_for_display(image), cmap="inferno")
    fx, fy = fovea_xy

    if exclusion_mask is not None and exclusion_mask.any():
        rgba = np.zeros((*exclusion_mask.shape, 4), dtype=np.float32)
        rgba[..., 0] = 1.0
        rgba[..., 3] = 0.25 * exclusion_mask.astype(np.float32)
        ax.imshow(rgba)

    all_radii = [radii[0], radii[1], radii[2], *extra_radii]
    for r in all_radii:
        ax.add_patch(Circle((fx, fy), r, fill=False, lw=1.1, color="white"))

    rmax = max(all_radii) if all_radii else max(image.shape)
    for ang_deg in (45, 135, 225, 315):
        ang = np.deg2rad(ang_deg)
        dx = rmax * np.cos(ang)
        dy = -rmax * np.sin(ang)
        ax.plot([fx, fx + dx], [fy, fy + dy], color="white", lw=0.9)

    ax.scatter([fx], [fy], s=18, c="cyan", marker="x")
    if title:
        ax.set_title(title, fontsize=9)
    ax.axis("off")


def _chunk_list(items, chunk_size: int):
    items = list(items)
    for start in range(0, len(items), chunk_size):
        yield items[start:start + chunk_size]


def save_overlay_mosaic_pdf(
    out_path: Path,
    case_id: str,
    transformed_maps: dict[str, np.ndarray],
    show_features: list[str],
    fovea_xy: tuple[float, float],
    radii: tuple[float, float, float],
    extra_radii: tuple[float, ...],
    exclusion_mask: np.ndarray,
    max_panels_per_page: int = 60,
    ncols: int = 3,
):
    names = [f for f in show_features if f in transformed_maps]
    if not names:
        names = list(transformed_maps)[:6]

    n_pages = int(np.ceil(len(names) / max_panels_per_page)) if names else 1

    with PdfPages(out_path) as pdf:
        for page_idx, page_names in enumerate(_chunk_list(names, max_panels_per_page), start=1):
            n = len(page_names)
            if n == 0:
                continue

            this_ncols = min(ncols, n)
            nrows = int(np.ceil(n / this_ncols))
            fig, axes = plt.subplots(
                nrows,
                this_ncols,
                figsize=(4.2 * this_ncols, 4.2 * nrows),
                dpi=180,
            )
            axes = np.atleast_1d(axes).ravel()

            for ax, name in zip(axes, page_names):
                draw_etdrs_overlay(
                    ax,
                    transformed_maps[name],
                    fovea_xy=fovea_xy,
                    radii=radii,
                    extra_radii=extra_radii,
                    exclusion_mask=exclusion_mask,
                    title=name,
                )

            for ax in axes[n:]:
                ax.axis("off")

            fig.suptitle(f"{case_id} | overlay page {page_idx}/{n_pages}", fontsize=11)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def save_summary_plots(
    out_dir: Path,
    case_id: str,
    feature_name: str,
    feature_image: np.ndarray,
    masks: dict[str, np.ndarray],
    region_means: dict[str, float],
    exclusion_mask: np.ndarray,
):
    ring_order = ["center", "inner_ring", "outer_ring"]
    extra_ring_names = sorted(k for k in masks if k.startswith("extra_ring_") and k.count("_") == 2)
    ring_order.extend(extra_ring_names)
    ring_order.extend(["outer_region", "whole"])

    ring_vals = [region_means.get(k, np.nan) for k in ring_order]
    finite = np.array([v for v in region_means.values() if np.isfinite(v)], dtype=np.float32)

    if finite.size:
        vmin = float(np.nanpercentile(finite, 2))
        vmax = float(np.nanpercentile(finite, 98))
        if vmax <= vmin:
            vmax = vmin + 1e-6
    else:
        vmin, vmax = 0.0, 1.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=180)

    bg = _normalize_for_display(feature_image)
    axes[0].imshow(bg, cmap="gray", alpha=0.28)
    region_map = build_region_value_map(masks, region_means)
    show = np.ma.masked_invalid(region_map)
    im = axes[0].imshow(show, cmap="viridis", vmin=vmin, vmax=vmax, alpha=0.95)

    if exclusion_mask is not None and exclusion_mask.any():
        rgba = np.zeros((*exclusion_mask.shape, 4), dtype=np.float32)
        rgba[..., 0] = 1.0
        rgba[..., 3] = 0.16 * exclusion_mask.astype(np.float32)
        axes[0].imshow(rgba)

    axes[0].axis("off")
    axes[0].set_title(f"{feature_name}: ETDRS region heatmap", fontsize=9)
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    axes[1].bar(np.arange(len(ring_order)), ring_vals)
    axes[1].set_xticks(np.arange(len(ring_order)))
    axes[1].set_xticklabels(ring_order, rotation=45, ha="right")
    axes[1].set_title(f"{feature_name}: whole-ring means")
    axes[1].set_ylabel("mean")

    fig.suptitle(case_id)
    fig.tight_layout()
    fig.savefig(out_dir / f"{case_id}__{feature_name}__summary.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------

def process_case(
    case_id: str,
    enface_path: Path,
    annotation_root: Path,
    out_dir: Path,
    onh_label: int,
    fovea_label: int,
    radii: tuple[float, float, float],
    extra_radii: tuple[float, ...],
    show_features: list[str] | None,
    max_panels_per_page: int = 60,
    save_mosaic: bool = False,
    save_summaries: bool = False,
) -> pd.DataFrame:
    maps = load_enface_maps(enface_path)
    first_map = next(iter(maps.values()))
    shape = first_map.shape

    if show_features is None:
        show_features = list(maps)[:6]
    else:
        show_features = [f for f in show_features if f in maps]

    ann_path = resolve_annotation_path(case_id, annotation_root)
    ann = load_annotation_enface(ann_path)
    ann = make_enface_isotropic_x(ann, x_scale=2.0, order=0)

    onh_mask = ann == onh_label
    fovea_mask = ann == fovea_label
    if onh_mask.sum() == 0 or fovea_mask.sum() == 0:
        raise ValueError(f"Missing ONH or fovea label in {ann_path}")

    onh_xy = centroid_from_mask(onh_mask)
    fovea_xy = centroid_from_mask(fovea_mask)
    exclusion_orig = build_onh_and_lateral_exclusion(onh_mask, fovea_xy=fovea_xy)

    eye = parse_eye_from_name(case_id)
    pts = {"fovea": fovea_xy, "onh": onh_xy}

    if eye == "OS":
        onh_mask, pts_flipped = flip_lr_image_and_points(onh_mask.astype(np.float32), pts)
        onh_mask = onh_mask > 0.5
        exclusion_orig, _ = flip_lr_image_and_points(exclusion_orig.astype(np.float32), pts)
        exclusion_orig = exclusion_orig > 0.5
        fovea_xy = pts_flipped["fovea"]
        onh_xy = pts_flipped["onh"]
    else:
        fovea_xy = pts["fovea"]
        onh_xy = pts["onh"]

    A_ref, pts_std, _ = build_rigid_standardize_transform(
        input_shape=shape,
        fovea_xy=fovea_xy,
        onh_xy=onh_xy,
        output_shape=shape,
    )
    exclusion_std = warp_mask(exclusion_orig, A_ref, output_shape=shape)
    fovea_std_xy = pts_std["fovea"]

    masks = make_etdrs_grid_plus_rings(
        shape=shape,
        fovea_xy=fovea_std_xy,
        onh_xy=pts_std["onh"],
        eye="R",
        radii=radii,
        extra_radii=extra_radii,
    )
    masks = add_sectorized_ring_masks(masks, fovea_xy=fovea_std_xy)
    masks = apply_exclusion_to_masks(masks, exclusion_std)

    overlay_maps: dict[str, np.ndarray] = {}
    summary_payloads: dict[str, tuple[np.ndarray, dict[str, float]]] = {}
    rows: list[dict] = []

    for feature_name, arr in maps.items():
        if eye == "OS":
            arr = np.fliplr(arr)

        arr_std = warp_image(arr, A_ref, output_shape=shape, order=1, cval=np.nan)
        region_means_full = summarize_by_regions(arr_std, masks, stats=("mean",))
        region_means = {k.replace("__mean", ""): v for k, v in region_means_full.items()}

        for region_name, value in region_means.items():
            rows.append(
                {
                    "case_id": case_id,
                    "feature": feature_name,
                    "region": region_name,
                    "stat": "mean",
                    "value": value,
                }
            )

        if feature_name in show_features:
            if save_mosaic:
                overlay_maps[feature_name] = arr_std
            if save_summaries:
                summary_payloads[feature_name] = (arr_std, region_means)

    if save_mosaic and overlay_maps:
        overlay_dir = out_dir / "overlays"
        overlay_dir.mkdir(parents=True, exist_ok=True)
        save_overlay_mosaic_pdf(
            overlay_dir / f"{case_id}__overlay.pdf",
            case_id=case_id,
            transformed_maps=overlay_maps,
            show_features=show_features,
            fovea_xy=fovea_std_xy,
            radii=radii,
            extra_radii=extra_radii,
            exclusion_mask=exclusion_std,
            max_panels_per_page=max_panels_per_page,
        )

    if save_summaries and summary_payloads:
        summary_plot_dir = out_dir / "summary_plots"
        summary_plot_dir.mkdir(parents=True, exist_ok=True)
        for feature_name, (feature_image, region_means) in summary_payloads.items():
            save_summary_plots(
                summary_plot_dir,
                case_id=case_id,
                feature_name=feature_name,
                feature_image=feature_image,
                masks=masks,
                region_means=region_means,
                exclusion_mask=exclusion_std,
            )

    return pd.DataFrame(rows)


def _run_one_case(task: dict) -> pd.DataFrame:
    return process_case(**task)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--enface-root", type=Path, required=True)
    ap.add_argument("--annotation-root", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--glob", default="*.npz", help="Case discovery glob under --enface-root")
    ap.add_argument("--onh-label", type=int, default=1)
    ap.add_argument("--fovea-label", type=int, default=2)
    ap.add_argument("--max-panels-per-page", type=int, default=60)
    ap.add_argument(
        "--radii",
        type=float,
        nargs=3,
        default=(75.0, 225.0, 450.0),
        help="center inner outer radii in enface pixels",
    )
    ap.add_argument(
        "--extra-radii",
        type=float,
        nargs="*",
        default=(600.0, 750.0),
        help="extra outer ring radii in enface pixels",
    )
    ap.add_argument(
        "--show-features",
        nargs="*",
        default=None,
        help="Feature names to overlay / plot. Default: first six found in each case.",
    )
    ap.add_argument("--save_mosaic", action="store_true")
    ap.add_argument("--save_summaries", action="store_true")
    ap.add_argument("--n_jobs", type=int, default=1)
    ap.add_argument("--fail_fast", action="store_true")

    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    cases = discover_cases(args.enface_root, args.glob)
    print(f"found {len(cases)} cases")

    tasks = [
        dict(
            case_id=case_id,
            enface_path=enface_path,
            annotation_root=args.annotation_root,
            out_dir=args.outdir,
            onh_label=args.onh_label,
            fovea_label=args.fovea_label,
            radii=tuple(args.radii),
            extra_radii=tuple(args.extra_radii),
            show_features=args.show_features,
            max_panels_per_page=args.max_panels_per_page,
            save_mosaic=args.save_mosaic,
            save_summaries=args.save_summaries,
        )
        for case_id, enface_path in cases
    ]

    all_df: list[pd.DataFrame] = []
    failures: list[tuple[str, str]] = []

    if args.n_jobs <= 1:
        for task in tasks:
            case_id = task["case_id"]
            print(f"starting {case_id}")
            try:
                df = _run_one_case(task)
                all_df.append(df)
                print(f"finished {case_id}")
            except Exception as e:
                if args.fail_fast:
                    raise
                failures.append((case_id, repr(e)))
                print(f"FAILED {case_id}: {e}")
    else:
        max_workers = min(args.n_jobs, len(tasks))
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            future_to_case = {ex.submit(_run_one_case, task): task["case_id"] for task in tasks}
            for fut in as_completed(future_to_case):
                case_id = future_to_case[fut]
                try:
                    df = fut.result()
                    all_df.append(df)
                    print(f"finished {case_id}")
                except Exception as e:
                    if args.fail_fast:
                        raise
                    failures.append((case_id, repr(e)))
                    print(f"FAILED {case_id}: {e}")

    if not all_df:
        raise RuntimeError("No cases completed successfully")

    summary_long = pd.concat(all_df, ignore_index=True)
    summary_long.to_csv(args.outdir / "texture_region_summary_long.csv", index=False)

    summary_wide = summary_long.pivot_table(
        index="case_id",
        columns=["feature", "region", "stat"],
        values="value",
    )
    summary_wide.to_csv(args.outdir / "texture_region_summary_wide.csv")

    if failures:
        fail_df = pd.DataFrame(failures, columns=["case_id", "error"])
        fail_df.to_csv(args.outdir / "failed_cases.csv", index=False)
        print(f"wrote failures for {len(failures)} cases")


if __name__ == "__main__":
    main()
