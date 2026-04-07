from __future__ import annotations
import json
from pathlib import Path
from skimage.transform import resize


import numpy as np

import code_files.file_utils as fu
from code_files.texture_package_prod.texture_extraction_utilities import (
    compute_dense_texture_maps,
    resample_map_to_image,
)

from pathlib import Path

import code_files.zarr_file_utils as zfu

def _reduce_between_bounds(
    volume,
    upper,
    lower,
    stat="mean",
    interp_x=True,
    extra_pad=2,
):
    """
    Reduce a flattened (Z,Y,X) volume into a (Z,X) en-face map
    between upper and lower Y-bounds.
    """
    if stat == "mean":
        projector = TextureSlabProjector(
            texture_vol=volume,
            reference_line=(np.asarray(upper) + np.asarray(lower)) / 2,
            max_top_offset=0,
            max_bottom_offset=0,
            extra_pad=extra_pad,
        )
        return projector.project_between(upper, lower, interp_x=interp_x)

    if stat not in {"median", "std"}:
        raise ValueError("stat must be 'mean', 'median', or 'std'")

    vol = np.asarray(volume, dtype=np.float32)
    upper = np.asarray(upper, dtype=np.float32)
    lower = np.asarray(lower, dtype=np.float32)

    top = np.floor(np.minimum(upper, lower)).astype(np.int32)
    bottom = np.ceil(np.maximum(upper, lower)).astype(np.int32)

    y0 = max(0, int(np.nanmin(top)) - extra_pad)
    y1 = min(vol.shape[1] - 1, int(np.nanmax(bottom)) + extra_pad)

    crop = vol[:, y0:y1 + 1, :]
    top = np.clip(top - y0, 0, crop.shape[1] - 1)
    bottom = np.clip(bottom - y0, 0, crop.shape[1] - 1)

    out = np.full((crop.shape[0], crop.shape[2]), np.nan, dtype=np.float32)

    for z in range(crop.shape[0]):
        for x in range(crop.shape[2]):
            lo = min(top[z, x], bottom[z, x])
            hi = max(top[z, x], bottom[z, x]) + 1
            vals = crop[z, lo:hi, x]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            out[z, x] = np.nanmedian(vals) if stat == "median" else np.nanstd(vals)

    if interp_x:
        for z in range(out.shape[0]):
            good = np.isfinite(out[z])
            if good.sum() >= 2:
                out[z] = np.interp(np.arange(out.shape[1]), np.where(good)[0], out[z, good])
            elif good.sum() == 1:
                out[z, :] = out[z, good][0]

    return out


def slab_stat_map(
    flat_volume,
    reference_line,
    bottom_offset,
    top_offset,
    stat="mean",
    interp_x=True,
):
    """
    Positive offsets are upward in the image.
    """
    ref = np.asarray(reference_line, dtype=np.float32)
    upper = ref - top_offset
    lower = ref - bottom_offset

    return _reduce_between_bounds(
        volume=flat_volume,
        upper=upper,
        lower=lower,
        stat=stat,
        interp_x=interp_x,
    )


def full_retina_stat_map(
    flat_volume,
    ilm_line,
    rpe_line,
    stat="mean",
    interp_x=True,
):
    return _reduce_between_bounds(
        volume=flat_volume,
        upper=ilm_line,
        lower=rpe_line,
        stat=stat,
        interp_x=interp_x,
    )


def retinal_thickness_map(
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    abs_value: bool = True,
) -> np.ndarray:
    """
    Return a (Z, X) thickness map from two surfaces.
    For current use this is usually ilm_smooth minus the algorithm-selected RPE-family line.
    """
    upper = np.asarray(upper_bound, dtype=np.float32)
    lower = np.asarray(lower_bound, dtype=np.float32)

    if upper.shape != lower.shape:
        raise ValueError('upper_bound and lower_bound must have the same shape')

    out = upper - lower
    if abs_value:
        out = np.abs(out)
    return out.astype(np.float32, copy=False)


def compute_extra_enface_maps(
    flat_volume,
    flat_layers,
    reference_key="rpe_smooth",
    slab_offsets=((5, 15),(5,30)),
    interp_x=True,
):
    """
    Returns direct en-face maps from a flattened image volume.
    """
    out = {}

    rpe = np.asarray(flat_layers[reference_key], dtype=np.float32)
    ilm = np.asarray(flat_layers["ilm_smooth"], dtype=np.float32)

    out["retinal_thickness"] = retinal_thickness_map(ilm, rpe, abs_value=True)

    for bottom, top in slab_offsets:
        # for stat in ("mean", "median", "std"):
        for stat in (["mean"]):
            out[f"slab_{stat}|{bottom}->{top}"] = slab_stat_map(
                flat_volume=flat_volume,
                reference_line=rpe,
                bottom_offset=bottom,
                top_offset=top,
                stat=stat,
                interp_x=interp_x,
            )

    # for stat in ("mean", "median", "std"):
    for stat in (["mean"]):
        out[f"full_retina_{stat}"] = full_retina_stat_map(
            flat_volume=flat_volume,
            ilm_line=ilm,
            rpe_line=rpe,
            stat=stat,
            interp_x=interp_x,
        )

    return out


def compute_textures_on_enface_maps(
    enface_maps,
    window=31,
    step=4,
    families=("firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm", "lbp", "gradient"),
    include_wavelet=False,
    levels=16,
    min_valid_frac=0.5,
    n_jobs=8,
):
    """
    Run the usual texture family on each 2D en-face map.
    """
    out = {}

    for map_name, arr in enface_maps.items():
        arr = np.asarray(arr, dtype=np.float32)
        mask = np.isfinite(arr)

        feat_maps, meta = compute_dense_texture_maps(
            image=arr,
            window=window,
            step=step,
            mask=mask,
            families=families,
            include_wavelet=include_wavelet,
            levels=levels,
            min_valid_frac=min_valid_frac,
            n_jobs=n_jobs,
        )

        if step != 1:
            feat_maps = {
                feat_name: resample_map_to_image(feat_map, meta)
                for feat_name, feat_map in feat_maps.items()
            }

        for feat_name, feat_map in feat_maps.items():
            out[f"{map_name}__{feat_name}"] = np.asarray(feat_map, dtype=np.float32)


    return out


class CompactTextureSlabProjector:
    """
    Compact (Z, R, C) projector.

    mean/std:
        vectorized via prefix sums

    median:
        exact fallback with loops
    """
    def __init__(
        self,
        texture_vol,                  # (Z, R, C), ndarray or zarr-backed
        row_centers_full: np.ndarray,
        col_centers: np.ndarray,
    ):
        if getattr(texture_vol, "ndim", None) != 3:
            raise ValueError("texture_vol must be (Z, R, C)")

        self.texture_vol = texture_vol
        self.row_centers_full = np.asarray(row_centers_full, dtype=np.int32)
        self.col_centers = np.asarray(col_centers, dtype=np.int32)

        self.Z, self.R, self.C = texture_vol.shape

        if len(self.row_centers_full) != self.R:
            raise ValueError("row_centers_full length does not match texture_vol.shape[1]")
        if len(self.col_centers) != self.C:
            raise ValueError("col_centers length does not match texture_vol.shape[2]")

    def _sample_bounds(self, upper_bound, lower_bound):
        upper_s = np.asarray(upper_bound, dtype=np.float32)[:, self.col_centers]
        lower_s = np.asarray(lower_bound, dtype=np.float32)[:, self.col_centers]

        top = np.minimum(upper_s, lower_s)
        bottom = np.maximum(upper_s, lower_s)

        start = np.searchsorted(self.row_centers_full, top, side="left")
        stop = np.searchsorted(self.row_centers_full, bottom, side="right")

        start = np.clip(start, 0, self.R)
        stop = np.clip(stop, 0, self.R)
        return start, stop

    def _project_mean_or_std(self, start, stop, stat="mean"):
        out = np.full((self.Z, self.C), np.nan, dtype=np.float32)

        c_idx = np.arange(self.C)[None, :]

        for z in range(self.Z):
            arr = np.asarray(self.texture_vol[z], dtype=np.float32)  # (R, C)

            valid = np.isfinite(arr)
            vals = np.where(valid, arr, 0.0)
            vals2 = vals * vals

            csum = np.empty((self.R + 1, self.C), dtype=np.float32)
            csum2 = np.empty((self.R + 1, self.C), dtype=np.float32)
            ccnt = np.empty((self.R + 1, self.C), dtype=np.int32)

            csum[0] = 0
            csum2[0] = 0
            ccnt[0] = 0

            np.cumsum(vals, axis=0, out=csum[1:])
            np.cumsum(vals2, axis=0, out=csum2[1:])
            np.cumsum(valid.astype(np.int32), axis=0, out=ccnt[1:])

            s0 = start[z]
            s1 = stop[z]

            sums = csum[s1, c_idx[0]] - csum[s0, c_idx[0]]
            cnts = ccnt[s1, c_idx[0]] - ccnt[s0, c_idx[0]]

            good = cnts > 0
            if stat == "mean":
                out[z, good] = sums[good] / cnts[good]
            else:
                sums2 = csum2[s1, c_idx[0]] - csum2[s0, c_idx[0]]
                mean = np.zeros_like(sums, dtype=np.float32)
                mean[good] = sums[good] / cnts[good]
                var = np.zeros_like(sums, dtype=np.float32)
                var[good] = sums2[good] / cnts[good] - mean[good] ** 2
                var = np.maximum(var, 0.0)
                out[z, good] = np.sqrt(var[good])

        return out

    def _project_median(self, start, stop):
        out = np.full((self.Z, self.C), np.nan, dtype=np.float32)

        for z in range(self.Z):
            arr = np.asarray(self.texture_vol[z], dtype=np.float32)
            for c in range(self.C):
                lo = int(start[z, c])
                hi = int(stop[z, c])
                if hi <= lo:
                    continue
                vals = arr[lo:hi, c]
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    continue
                out[z, c] = np.nanmedian(vals)

        return out

    def project_between(
        self,
        upper_bound,
        lower_bound,
        stat="mean",
    ):
        start, stop = self._sample_bounds(upper_bound, lower_bound)

        if stat == "mean":
            return self._project_mean_or_std(start, stop, stat="mean")
        if stat == "std":
            return self._project_mean_or_std(start, stop, stat="std")
        if stat == "median":
            return self._project_median(start, stop)

        raise ValueError("stat must be 'mean', 'median', or 'std'")

    def upsample_x(self, sampled_map, full_x):
        sampled_map = np.asarray(sampled_map, dtype=np.float32)
        out = np.full((self.Z, full_x), np.nan, dtype=np.float32)

        for z in range(self.Z):
            good = np.isfinite(sampled_map[z])
            if good.sum() >= 2:
                out[z] = np.interp(
                    np.arange(full_x),
                    self.col_centers[good],
                    sampled_map[z, good],
                )
            elif good.sum() == 1:
                out[z, :] = sampled_map[z, good][0]

        return out

from joblib import Parallel, delayed

def _project_one_compact_feature_from_zarr(
    compact_zarr_path,
    feat,
    center,
    row_centers_full,
    col_centers,
    candidate_slabs,
    stat,
    upsample,
):
    import zarr
    import numpy as np

    root = zarr.open_group(str(compact_zarr_path), mode="r")

    proj = CompactTextureSlabProjector(
        texture_vol=root[feat],
        row_centers_full=row_centers_full,
        col_centers=col_centers,
    )

    full_x = center.shape[1]
    out = {}

    for bottom_offset, top_offset in candidate_slabs:
        sampled = proj.project_between(
            upper_bound=center - top_offset,
            lower_bound=center - bottom_offset,
            stat=stat,
        )

        arr = proj.upsample_x(sampled, full_x=full_x) if upsample else sampled
        out[f"{feat}|{bottom_offset}->{top_offset}|{stat}"] = arr

    return out

def project_texture_compact_zarr_to_enface_for_volume(
    vol_path,
    compact_zarr_path,
    flat_layers_npz,
    line_key=None,
    candidate_slabs=((5, 15),),
    features=None,
    stat="mean",
    upsample=True,
    n_jobs=1,
):
    import json
    import zarr
    import numpy as np
    from pathlib import Path
    from joblib import Parallel, delayed
    from code_files import file_utils

    vol_path = Path(vol_path)
    compact_zarr_path = Path(compact_zarr_path)

    root = zarr.open_group(str(compact_zarr_path), mode="r")

    manifest_json = root.attrs.get("manifest_json", None)
    if manifest_json is None:
        raise ValueError(f"No manifest_json found in {compact_zarr_path}")
    manifest = json.loads(manifest_json)

    flat_layers = np.load(flat_layers_npz)

    if line_key is None:
        line_key = file_utils.get_algorithm_key_from_filepath(vol_path)

    center = np.asarray(flat_layers[line_key], dtype=np.float32)
    row_centers_full = np.asarray(manifest["row_centers_full"], dtype=np.int32)
    col_centers = np.asarray(manifest["col_centers"], dtype=np.int32)

    if features is None:
        features = sorted(root.array_keys())

    if n_jobs == 1:
        pieces = [
            _project_one_compact_feature_from_zarr(
                compact_zarr_path=compact_zarr_path,
                feat=feat,
                center=center,
                row_centers_full=row_centers_full,
                col_centers=col_centers,
                candidate_slabs=candidate_slabs,
                stat=stat,
                upsample=upsample,
            )
            for feat in features
        ]
    else:
        pieces = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_project_one_compact_feature_from_zarr)(
                compact_zarr_path=compact_zarr_path,
                feat=feat,
                center=center,
                row_centers_full=row_centers_full,
                col_centers=col_centers,
                candidate_slabs=candidate_slabs,
                stat=stat,
                upsample=upsample,
            )
            for feat in features
        )

    out = {}
    for d in pieces:
        out.update(d)
    return out


class TextureSlabProjector:
    def __init__(
        self,
        texture_vol,
        reference_line,
        max_top_offset,
        max_bottom_offset,
        extra_pad=0,
    ):
        import numpy as np

        ref = np.asarray(reference_line, dtype=np.float32)   # (Z, X)

        top = np.floor(ref - max_top_offset).astype(np.int32)
        bottom = np.ceil(ref - max_bottom_offset).astype(np.int32)

        y_min = int(np.nanmin(np.minimum(top, bottom))) - extra_pad
        y_max = int(np.nanmax(np.maximum(top, bottom))) + extra_pad

        full_Y = texture_vol.shape[1]
        y_min = max(0, y_min)
        y_max = min(full_Y - 1, y_max)

        arr = np.asarray(texture_vol[:, y_min:y_max + 1, :], dtype=np.float32)
        self.y0 = y_min
        self.Z, self.Y, self.X = arr.shape

        valid = np.isfinite(arr)
        vals = np.where(valid, arr, 0.0)

        vals_yzx = np.transpose(vals, (1, 0, 2))
        cnts_yzx = np.transpose(valid.astype(np.int32), (1, 0, 2))

        self.csum = np.empty((self.Y + 1, self.Z, self.X), dtype=np.float32)
        self.ccnt = np.empty((self.Y + 1, self.Z, self.X), dtype=np.int32)

        self.csum[0] = 0
        self.ccnt[0] = 0
        np.cumsum(vals_yzx, axis=0, out=self.csum[1:])
        np.cumsum(cnts_yzx, axis=0, out=self.ccnt[1:])

    def project_between(self, upper_bound, lower_bound, interp_x=True):
        import numpy as np

        top = np.floor(np.minimum(upper_bound, lower_bound)).astype(np.int32)
        bottom = np.ceil(np.maximum(upper_bound, lower_bound)).astype(np.int32)

        # move from full-image Y coords into cropped coords
        top = top - self.y0
        bottom = bottom - self.y0

        # example here: exclude the boundary rows themselves
        start = top + 1
        stop = bottom

        start = np.clip(start, 0, self.Y)
        stop = np.clip(stop, 0, self.Y)

        z_idx = np.arange(self.Z)[:, None]
        x_idx = np.arange(self.X)[None, :]

        sums = self.csum[stop, z_idx, x_idx] - self.csum[start, z_idx, x_idx]
        cnts = self.ccnt[stop, z_idx, x_idx] - self.ccnt[start, z_idx, x_idx]

        out = np.full((self.Z, self.X), np.nan, dtype=np.float32)
        good = cnts > 0
        out[good] = sums[good] / cnts[good]

        if interp_x:
            for z in range(self.Z):
                g = np.isfinite(out[z])
                if g.sum() >= 2:
                    out[z] = np.interp(np.arange(self.X), np.where(g)[0], out[z, g])

        return out




def make_enface_isotropic_x(
    arr: np.ndarray,
    x_scale: float = 2.0,
    order: int = 1,
) -> np.ndarray:
    """
    Resample a (Z, X) en-face map so pixels become isotropic by scaling X only.

    For your current maps, use x_scale=2.0:
        (1024, 512) -> (1024, 1024)

    order:
      1 for continuous feature maps
      0 for masks / labels
    """
    arr = np.asarray(arr)
    out_x = int(round(arr.shape[1] * float(x_scale)))
    out = resize(
        arr.astype(np.float32, copy=False),
        (arr.shape[0], out_x),
        order=order,
        preserve_range=True,
        anti_aliasing=(order > 0),
    )
    if order == 0:
        return out > 0.5 if arr.dtype == bool else np.rint(out).astype(arr.dtype, copy=False)
    return out.astype(np.float32, copy=False)




##
import json
from pathlib import Path

import numpy as np
import zarr


def _load_compact_manifest(compact_zarr_path: str | Path) -> dict:
    root = zarr.open_group(str(compact_zarr_path), mode="r")
    manifest_json = root.attrs.get("manifest_json", None)
    if manifest_json is None:
        raise ValueError(f"No manifest_json found in {compact_zarr_path}")
    return json.loads(manifest_json)


def _interp_along_x_to_full_width(
    arr_zx: np.ndarray,
    col_centers: np.ndarray,
    full_x: int,
) -> np.ndarray:
    """
    Input arr_zx shape:
      (Zsmall, Csampled), where columns correspond to manifest['col_centers'].
    Output:
      (Zsmall, full_x)
    """
    arr_zx = np.asarray(arr_zx, dtype=np.float32)
    out = np.full((arr_zx.shape[0], full_x), np.nan, dtype=np.float32)

    xx = np.arange(full_x, dtype=np.float32)
    col_centers = np.asarray(col_centers, dtype=np.float32)

    for z in range(arr_zx.shape[0]):
        good = np.isfinite(arr_zx[z])
        if good.sum() >= 2:
            out[z] = np.interp(xx, col_centers[good], arr_zx[z, good])
        elif good.sum() == 1:
            out[z, :] = arr_zx[z, good][0]

    return out


def _interp_along_z_to_full_depth(
    arr_zx: np.ndarray,
    z_indices: np.ndarray,
    full_z: int,
) -> np.ndarray:
    """
    Input arr_zx shape:
      (Zsampled, Xfull_or_sampled), where rows correspond to manifest['z_indices'].
    Output:
      (full_z, X)
    """
    arr_zx = np.asarray(arr_zx, dtype=np.float32)
    out = np.full((full_z, arr_zx.shape[1]), np.nan, dtype=np.float32)

    zz = np.arange(full_z, dtype=np.float32)
    z_indices = np.asarray(z_indices, dtype=np.float32)

    for x in range(arr_zx.shape[1]):
        good = np.isfinite(arr_zx[:, x])
        if good.sum() >= 2:
            out[:, x] = np.interp(zz, z_indices[good], arr_zx[good, x])
        elif good.sum() == 1:
            out[:, x] = arr_zx[good, x][0]

    return out


def upsample_projected_compact_enface_map(
    projected_map: np.ndarray,
    compact_zarr_path: str | Path | None = None,
    manifest: dict | None = None,
    full_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """
    Upsample an already-projected compact texture en-face map to full (Z, X),
    without instantiating the full texture volume zarr.

    Handles both cases:
      1) projected_map is still compact in X and Z
      2) projected_map is already full in X but compact in Z
         (this is what you usually get if projection used upsample=True)

    If full_shape is None, use manifest['volume_shape'] -> (full_z, full_x).
    """
    if manifest is None:
        if compact_zarr_path is None:
            raise ValueError("Provide either manifest or compact_zarr_path")
        manifest = _load_compact_manifest(compact_zarr_path)

    arr = np.asarray(projected_map, dtype=np.float32)

    vol_shape = tuple(int(v) for v in manifest["volume_shape"])
    full_z = int(vol_shape[0])
    full_x = int(vol_shape[2])

    if full_shape is not None:
        full_z, full_x = map(int, full_shape)

    z_indices = np.asarray(manifest["z_indices"], dtype=int)
    col_centers = np.asarray(manifest["col_centers"], dtype=int)

    # X upsample only if still sampled-grid width.
    if arr.shape[1] == len(col_centers) and full_x != len(col_centers):
        arr = _interp_along_x_to_full_width(arr, col_centers=col_centers, full_x=full_x)
    elif arr.shape[1] != full_x:
        raise ValueError(
            f"Projected map X size {arr.shape[1]} does not match either "
            f"len(col_centers)={len(col_centers)} or full_x={full_x}"
        )

    # Z upsample only if still compact in Z.
    if arr.shape[0] == len(z_indices) and full_z != len(z_indices):
        arr = _interp_along_z_to_full_depth(arr, z_indices=z_indices, full_z=full_z)
    elif arr.shape[0] != full_z:
        raise ValueError(
            f"Projected map Z size {arr.shape[0]} does not match either "
            f"len(z_indices)={len(z_indices)} or full_z={full_z}"
        )

    return arr.astype(np.float32, copy=False)


def upsample_projected_compact_enface_maps(
    projected_maps: dict[str, np.ndarray],
    compact_zarr_path: str | Path | None = None,
    manifest: dict | None = None,
    full_shape: tuple[int, int] | None = None,
) -> dict[str, np.ndarray]:
    if manifest is None:
        if compact_zarr_path is None:
            raise ValueError("Provide either manifest or compact_zarr_path")
        manifest = _load_compact_manifest(compact_zarr_path)

    return {
        k: upsample_projected_compact_enface_map(
            v,
            manifest=manifest,
            full_shape=full_shape,
        )
        for k, v in projected_maps.items()
    }


def _parse_csv_arg(text):
    if text is None or str(text).strip() == "":
        return None
    return tuple(x.strip() for x in str(text).split(",") if x.strip())


def _resolve_compact_texture_zarr_path(texture_root_dir, vol_path, texture_run=None):
    volume_dir = Path(texture_root_dir) / vol_path.stem
    if not volume_dir.exists():
        raise FileNotFoundError(f"Could not find texture volume dir: {volume_dir}")

    if texture_run is not None:
        zp = volume_dir / texture_run / "texture_bscan_maps_compact.zarr"
        if not zp.exists():
            raise FileNotFoundError(f"Could not find compact zarr for run {texture_run}: {zp}")
        return zp

    # txt_path = volume_dir / "compact_zarr_path.txt"
    # if txt_path.exists():
    #     zp = Path(txt_path.read_text().strip())
    #     if not zp.exists():
    #         raise FileNotFoundError(f"compact_zarr_path.txt points to missing file: {zp}")
    #     return zp

    run_paths = sorted(volume_dir.glob("*/texture_bscan_maps_compact.zarr"))
    if len(run_paths) == 1:
        return run_paths[0]
    if len(run_paths) == 0:
        raise FileNotFoundError(f"No compact texture zarr found under {volume_dir}")

    raise ValueError(
        f"Multiple compact texture runs found for {vol_path.stem}: "
        f"{[p.parent.name for p in run_paths]}. Pass --texture_run."
    )


def subset_dict(d, keys):
    return {k: d[k] for k in keys if k in d}


def prepare_one_volume_texture_enfaces(vol_path, args):
    vol_path = Path(vol_path)
    reference_key = fu.get_algorithm_key_from_filepath(vol_path)
    print(f"[{vol_path.stem}] using reference key: {reference_key}")

    art = zfu.ensure_flattened_artifacts(
        vol_path=vol_path,
        flatten_with=reference_key,
        layers_root=args.layers_root,
        z_stride=1,
        overwrite=args.overwrite_flatten,
        make_image_zarr=True,
        make_label_zarr=False,
        make_annotation_zarr=False,
        save_flat_layers_npz=True,
    )

    flat_volume = zarr.open_group(str(art["image_zarr"]), mode="r")["data"]
    flat_layers = np.load(art["flat_layers_npz"])

    extra_maps = compute_extra_enface_maps(
        flat_volume=flat_volume,
        flat_layers=flat_layers,
        reference_key=reference_key,
        slab_offsets=args.slab_offsets,
        interp_x=True,
    )

    texture_input_keys = [f"slab_mean|{b}->{t}" for b, t in args.slab_offsets]
    if args.include_full_retina:
        texture_input_keys.append("full_retina_mean")
    texture_input_maps = subset_dict(extra_maps, texture_input_keys)

    texture_maps = compute_textures_on_enface_maps(
        enface_maps=texture_input_maps,
        window=args.texture_window,
        step=args.texture_step,
        include_wavelet=False,
        levels=args.texture_levels,
        n_jobs=args.texture_n_jobs,
    )

    projected_texture_maps = {}
    if args.texture_root_dir is not None:
        compact_zarr_path = _resolve_compact_texture_zarr_path(
            texture_root_dir=args.texture_root_dir,
            vol_path=vol_path,
            texture_run=args.texture_run,
        )

        projected_texture_features = _parse_csv_arg(args.projected_texture_features)

        projected_texture_maps = project_texture_compact_zarr_to_enface_for_volume(
            vol_path=vol_path,
            compact_zarr_path=compact_zarr_path,
            flat_layers_npz=art["flat_layers_npz"],
            line_key=reference_key,
            candidate_slabs=args.slab_offsets,
            features=projected_texture_features,
            stat=args.projected_texture_stat,
            upsample=True,
            n_jobs=args.project_texture_n_jobs,
        )

    # After projected_texture_maps are created:
    if args.texture_root_dir is not None:
        projected_texture_maps = upsample_projected_compact_enface_maps(
            projected_texture_maps,
            compact_zarr_path=compact_zarr_path,
        )

    # Then make everything isotropic for downstream geometry:
    extra_maps = make_enface_dict_isotropic_x(extra_maps, x_scale=2.0, order=1)
    texture_maps = make_enface_dict_isotropic_x(texture_maps, x_scale=2.0, order=1)
    projected_texture_maps = make_enface_dict_isotropic_x(projected_texture_maps, x_scale=2.0, order=1)

    return dict(
        vol_path=vol_path,
        reference_key=reference_key,
        art=art,
        flat_volume=flat_volume,
        flat_layers=flat_layers,
        extra_maps=extra_maps,
        texture_maps=texture_maps,
        projected_texture_maps=projected_texture_maps,
        projected_texture_stat=args.projected_texture_stat,
    )


def make_enface_dict_isotropic_x(
    maps: dict[str, np.ndarray],
    x_scale: float = 2.0,
    order: int = 1,
) -> dict[str, np.ndarray]:
    return {
        k: make_enface_isotropic_x(v, x_scale=x_scale, order=order)
        for k, v in maps.items()
    }


def scale_point_xy_x(pt_xy: tuple[float, float], x_scale: float = 2.0) -> tuple[float, float]:
    x, y = pt_xy
    return float(x * x_scale), float(y)


def scale_points_dict_xy_x(
    points_xy: dict[str, tuple[float, float]],
    x_scale: float = 2.0,
) -> dict[str, tuple[float, float]]:
    return {k: scale_point_xy_x(v, x_scale=x_scale) for k, v in points_xy.items()}