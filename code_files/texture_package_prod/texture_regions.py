#Reviewed
from __future__ import annotations

import numpy as np
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle
from scipy import ndimage
from skimage.transform import AffineTransform, warp

from code_files.texture_package_prod.texture_enface_utils import make_enface_isotropic_x


def _rr_cc(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    return np.indices(shape)


# def make_etdrs_grid_plus_rings(
#     shape: tuple[int, int],
#     fovea_xy: tuple[float, float],
#     onh_xy: tuple[float, float] | None = None,
#     eye: str | None = None,
#     radii: tuple[float, float, float] | None = None,
#     extra_radii: tuple[float, ...] = (),
# ) -> dict[str, np.ndarray]:
#     """
#     Approximate ETDRS in pixel space: center + inner 4 sectors + outer 4 sectors.
#     Sector boundaries are diagonal lines, so sectors are centered on S/N/I/T.
#     Extra radii add wider fovea-centered circular rings beyond ETDRS.
#     """
#     h, w = shape
#     cx, cy = map(float, fovea_xy)
#     if radii is None:
#         outer = 0.18 * min(h, w)
#         inner = outer / 2
#         center = outer / 6
#     else:
#         center, inner, outer = map(float, radii)

#     if eye is None and onh_xy is not None:
#         eye = 'R' if onh_xy[0] > cx else 'L'
#     eye = 'R' if eye is None else eye.upper()

#     rr, cc = _rr_cc(shape)
#     dr = rr - cy
#     dc = cc - cx
#     rho = np.sqrt(dr * dr + dc * dc)
#     theta = (np.degrees(np.arctan2(-dr, dc)) + 360.0) % 360.0

#     right_sector = (theta < 45) | (theta >= 315)
#     superior_sector = (theta >= 45) & (theta < 135)
#     left_sector = (theta >= 135) & (theta < 225)
#     inferior_sector = (theta >= 225) & (theta < 315)

#     if eye == 'R':
#         nasal_sector = right_sector
#         temporal_sector = left_sector
#     else:
#         nasal_sector = left_sector
#         temporal_sector = right_sector

#     sectors = {
#         'superior': superior_sector,
#         'nasal': nasal_sector,
#         'inferior': inferior_sector,
#         'temporal': temporal_sector,
#     }

#     masks = {
#         'center': rho <= center,
#         'inner_ring': (rho > center) & (rho <= inner),
#         'outer_ring': (rho > inner) & (rho <= outer),
#     }
#     for ring_name in ('inner', 'outer'):
#         ring_mask = masks[f'{ring_name}_ring']
#         for sec_name, sec_mask in sectors.items():
#             masks[f'{ring_name}_{sec_name}'] = ring_mask & sec_mask

#     prev = outer
#     for i, rad in enumerate(sorted(extra_radii), start=1):
#         masks[f'extra_ring_{i}'] = (rho > prev) & (rho <= rad)
#         prev = rad
#     masks['whole'] = rho <= prev
#     return masks

def make_etdrs_grid_plus_rings(
    shape: tuple[int, int],
    fovea_xy: tuple[float, float],
    onh_xy: tuple[float, float] | None = None,
    eye: str | None = None,
    radii: tuple[float, float, float] | None = None,
    extra_radii: tuple[float, ...] = (),
) -> dict[str, np.ndarray]:
    """
    ETDRS center + inner/outer sector masks, optional wider rings, plus an
    unbounded outer region split by T/S/I/N outside the widest defined radius.
    """
    h, w = shape
    cx, cy = map(float, fovea_xy)

    center, inner, outer = map(float, radii)

    if eye is None and onh_xy is not None:
        eye = "R" if onh_xy[0] > cx else "L"
    eye = "R" if eye is None else eye.upper()

    rr, cc = np.indices(shape)
    dr = rr - cy
    dc = cc - cx
    rho = np.sqrt(dr * dr + dc * dc)
    theta = (np.degrees(np.arctan2(-dr, dc)) + 360.0) % 360.0

    right_sector    = (theta < 45) | (theta >= 315)
    superior_sector = (theta >= 45) & (theta < 135)
    left_sector     = (theta >= 135) & (theta < 225)
    inferior_sector = (theta >= 225) & (theta < 315)

    if eye == "R":
        nasal_sector = right_sector
        temporal_sector = left_sector
    else:
        nasal_sector = left_sector
        temporal_sector = right_sector

    sectors = {
        "temporal": temporal_sector,
        "superior": superior_sector,
        "nasal": nasal_sector,
        "inferior": inferior_sector,
    }

    masks = {
        "center": rho <= center,
        "inner_ring": (rho > center) & (rho <= inner),
        "outer_ring": (rho > inner) & (rho <= outer),
    }

    for ring_name in ("inner", "outer"):
        ring_mask = masks[f"{ring_name}_ring"]
        for sec_name, sec_mask in sectors.items():
            masks[f"{ring_name}_{sec_name}"] = ring_mask & sec_mask

    prev = outer
    for i, rad in enumerate(sorted(extra_radii), start=1):
        ring_mask = (rho > prev) & (rho <= rad)
        masks[f"extra_ring_{i}"] = ring_mask
        for sec_name, sec_mask in sectors.items():
            masks[f"extra_ring_{i}_{sec_name}"] = ring_mask & sec_mask
        prev = rad

    outer_region = rho > prev
    masks["outer_region"] = outer_region
    for sec_name, sec_mask in sectors.items():
        masks[f"outer_region_{sec_name}"] = outer_region & sec_mask

    masks["whole"] = np.ones(shape, dtype=bool)
    return masks

def make_etdrs_like_grid(*args, **kwargs) -> dict[str, np.ndarray]:
    """Backward-compatible alias."""
    return make_etdrs_grid_plus_rings(*args, **kwargs)


_STAT_FNS = {
    'mean': np.nanmean,
    'std': np.nanstd,
    'median': np.nanmedian,
    'p10': lambda x: np.nanpercentile(x, 10),
    'p90': lambda x: np.nanpercentile(x, 90),
}


def summarize_by_regions(
    feature_map: np.ndarray,
    masks: dict[str, np.ndarray],
    stats: tuple[str, ...] = ('mean', 'std', 'p90'),
) -> dict[str, float]:
    out = {}
    for name, mask in masks.items():
        vals = feature_map[mask]
        vals = vals[np.isfinite(vals)]
        for stat in stats:
            key = f'{name}__{stat}'
            out[key] = np.nan if vals.size == 0 else float(_STAT_FNS[stat](vals))
    return out


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

    ring_like = []

    # Only sectorize truly unsectorized extra rings
    for k in masks:
        if re.fullmatch(r"extra_ring_\d+", k):
            ring_like.append(k)

    # Optional: sectorize outer_region only if present and not already sectorized
    if "outer_region" in masks:
        already = any(f"outer_region_{sec}" in masks for sec in sectors)
        if not already:
            ring_like.append("outer_region")

    for ring_name in ring_like:
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
    # ax.imshow(_normalize_for_display(image), cmap="inferno")
    ax.imshow(image, cmap="inferno")
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



def prepare_case_for_etdrs(
    case_id: str,
    enface_path: Path,
    annotation_root: Path,
    onh_label: int,
    fovea_label: int,
    radii: tuple[float, float, float],
    extra_radii: tuple[float, ...],
) -> dict:
    case_id = canonical_case_id(case_id)
    enface_path = Path(enface_path)
    annotation_root = Path(annotation_root)

    maps = load_enface_maps(enface_path)
    first_map = next(iter(maps.values()))
    shape = first_map.shape

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
        exclusion_orig, pts_flipped = flip_lr_image_and_points(exclusion_orig.astype(np.float32), pts)
        exclusion_orig = exclusion_orig > 0.5
        maps = {k: np.fliplr(v) for k, v in maps.items()}
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

    maps_std = {
        feature_name: warp_image(arr, A_ref, output_shape=shape, order=1, cval=np.nan)
        for feature_name, arr in maps.items()
    }

    exclusion_std = warp_mask(exclusion_orig, A_ref, output_shape=shape)

    masks = make_etdrs_grid_plus_rings(
        shape=shape,
        fovea_xy=pts_std["fovea"],
        onh_xy=pts_std["onh"],
        eye="R",
        radii=radii,
        extra_radii=extra_radii,
    )
    masks = apply_exclusion_to_masks(masks, exclusion_std)

    return {
        "case_id": case_id,
        "eye": eye,
        "maps": maps_std,
        "masks": masks,
        "fovea_xy": pts_std["fovea"],
        "onh_xy": pts_std["onh"],
        "exclusion_mask": exclusion_std,
        "radii": tuple(radii),
        "extra_radii": tuple(extra_radii),
    }