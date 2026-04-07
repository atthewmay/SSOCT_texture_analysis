#Reviewed
from __future__ import annotations

import numpy as np


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
