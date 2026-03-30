
#REVIEWED!
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from itertools import product

import numpy as np
from joblib import Parallel, delayed
from scipy import ndimage
from scipy.stats import entropy as scipy_entropy
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.transform import resize
from skimage.measure import label as cc_label
import code_files.segmentation_code.segmentation_plot_utils as spu

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numcodecs
import zarr

# from .vessel_texture_postproc_utils import estimate_shadow_mask_from_bscan, postprocess_feature_dict


@dataclass
class DenseMapMeta:
    row_centers: np.ndarray
    col_centers: np.ndarray
    window: int
    step: int
    image_shape: tuple[int, int]


@dataclass(frozen=True)
class TextureSweepParams:
    """
    Same class used for:
      1. the public sweep definition (fields can be tuples/lists)
      2. the concrete single-run params passed into engine + per-B-scan worker
    """
    window: int | tuple[int, ...] = 31
    levels: int | tuple[int, ...] = 32
    gaussian_sigma: float | tuple[float, ...] = 0.0
    downsample_factor: int | tuple[int, ...] = 1

    @staticmethod
    def _to_tuple(value):
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, (list, tuple)):
            return tuple(value)
        return (value,)

    def iter_cases(self):
        windows = tuple(int(v) for v in self._to_tuple(self.window))
        levels = tuple(int(v) for v in self._to_tuple(self.levels))
        sigmas = tuple(float(v) for v in self._to_tuple(self.gaussian_sigma))
        downs = tuple(int(v) for v in self._to_tuple(self.downsample_factor))

        for window, levels, gaussian_sigma, downsample_factor in product(
            windows, levels, sigmas, downs
        ):
            yield TextureSweepParams(
                window=window,
                levels=levels,
                gaussian_sigma=gaussian_sigma,
                downsample_factor=downsample_factor,
            )

    def concrete(self) -> "TextureSweepParams":
        cases = list(self.iter_cases())
        if len(cases) != 1:
            raise ValueError(
                "TextureSweepParams must be concrete here. "
                "Pass a single case, not a sweep."
            )
        return cases[0]

    def tag(self) -> str:
        p = self.concrete()
        return (
            f"window={int(p.window)}__"
            f"levels={int(p.levels)}__"
            f"gaussian={float(p.gaussian_sigma):g}__"
            f"downsample={int(p.downsample_factor)}"
        )

    def as_attrs(self) -> dict:
        p = self.concrete()
        return {
            "window": int(p.window),
            "levels": int(p.levels),
            "gaussian_sigma": float(p.gaussian_sigma),
            "downsample_factor": int(p.downsample_factor),
        }


def preprocess_texture_image(
    image: np.ndarray,
    texture_params: TextureSweepParams,
) -> np.ndarray:
    """
    Apply Gaussian smoothing and optional downsample->upsample interpolation
    before texture extraction. Output remains same shape as input.
    """
    p = texture_params.concrete()
    img = np.asarray(image, dtype=np.float32)

    if float(p.gaussian_sigma) > 0:
        img = ndimage.gaussian_filter(img, sigma=float(p.gaussian_sigma))

    ds = int(p.downsample_factor)
    if ds > 1:
        small_shape = (
            max(1, int(np.round(img.shape[0] / ds))),
            max(1, int(np.round(img.shape[1] / ds))),
        )

        img_small = resize(
            img,
            small_shape,
            order=1,
            preserve_range=True,
            anti_aliasing=True,
        ).astype(np.float32, copy=False)

        img = resize(
            img_small,
            img.shape,
            order=1,
            preserve_range=True,
            anti_aliasing=False,
        ).astype(np.float32, copy=False)

    return img.astype(np.float32, copy=False)

GLCM_FEATURES = ('contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM')


def robust_rescale_uint(image: np.ndarray, levels: int = 32, clip_percentiles: tuple[float, float] = (1, 99)) -> np.ndarray:
    """Robustly rescale to integer gray levels."""
    img = image.astype(np.float32)
    lo, hi = np.nanpercentile(img, clip_percentiles)
    img = np.clip(img, lo, hi)
    if hi <= lo:
        return np.zeros_like(img, dtype=np.uint8)
    img = (img - lo) / (hi - lo)
    return np.clip(np.round(img * (levels - 1)), 0, levels - 1).astype(np.uint8)


# ---------- small patch features ----------

def _nanmean_std_percentiles(values: np.ndarray, bins: int = 32) -> dict[str, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            'mean': np.nan,
            'std': np.nan,
            'entropy': np.nan,
            'p10': np.nan,
            'p50': np.nan,
            'p90': np.nan,
        }
    hist, _ = np.histogram(values, bins=int(bins))
    probs = hist / max(hist.sum(), 1)
    return {
        'mean': float(np.nanmean(values)),
        'std': float(np.nanstd(values)),
        'entropy': float(scipy_entropy(probs + 1e-12, base=2)),
        'p10': float(np.nanpercentile(values, 10)),
        'p50': float(np.nanpercentile(values, 50)),
        'p90': float(np.nanpercentile(values, 90)),
    }

def glcm_features_patch(patch_q: np.ndarray, levels: int = 32) -> dict[str, float]:
    glcm = graycomatrix(
        patch_q,
        distances=[1], # EDIT? We may want longer range dependencies. The out would have to be named by the distnacees I think.
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=levels,
        symmetric=True,
        normed=True,
    )
    out = {f'glcm_{name}': float(graycoprops(glcm, name).mean()) for name in GLCM_FEATURES}
    p = glcm.astype(np.float64)
    out['glcm_entropy'] = float((-(p * np.log2(p + 1e-12))).sum(axis=(0, 1)).mean())
    return out


def _iter_lines(q: np.ndarray, angle: int):
    if angle == 0:
        for row in q:
            yield row
    elif angle == 90:
        for col in q.T:
            yield col
    elif angle == 45:
        for k in range(-q.shape[0] + 1, q.shape[1]):
            yield np.diagonal(np.fliplr(q), offset=k)
    elif angle == 135:
        for k in range(-q.shape[0] + 1, q.shape[1]):
            yield np.diagonal(q, offset=k)
    else:
        raise ValueError(angle)


def _run_length_matrix(q: np.ndarray, levels: int = 32, angles: tuple[int, ...] = (0, 45, 90, 135)) -> np.ndarray:
    """This function builds a GLRLM: a gray-level run-length matrix.

    What that means:

    rows = gray level
    columns = run length
    entry (g, r) = how many times you saw a run of gray level g with length r+1

    A “run” means consecutive equal-valued pixels along some direction."""
    max_run = max(q.shape)
    mats = []
    for angle in angles:
        mat = np.zeros((levels, max_run), dtype=np.float64)
        for line in _iter_lines(q, angle):
            if line.size == 0:
                continue
            start = 0
            for i in range(1, len(line) + 1):
                if i == len(line) or line[i] != line[start]:
                    gray = int(line[start])
                    run = i - start
                    mat[gray, run - 1] += 1
                    start = i
        mats.append(mat)
    return np.mean(mats, axis=0)


def glrlm_features_patch(patch_q: np.ndarray, levels: int = 32) -> dict[str, float]:
    p = _run_length_matrix(patch_q, levels=levels)
    total = p.sum()
    if total == 0:
        return {k: np.nan for k in ('glrlm_sre', 'glrlm_lre', 'glrlm_gln', 'glrlm_rln', 'glrlm_rp')}
    runs = np.arange(1, p.shape[1] + 1)[None, :]
    gray_sums = p.sum(axis=1)
    run_sums = p.sum(axis=0)
    return {
        'glrlm_sre': float((p / (runs * runs)).sum() / total),
        'glrlm_lre': float((p * (runs * runs)).sum() / total),
        'glrlm_gln': float((gray_sums * gray_sums).sum() / total),
        'glrlm_rln': float((run_sums * run_sums).sum() / total),
        'glrlm_rp': float(total / patch_q.size),
    }


def glszm_features_patch(patch_q: np.ndarray, levels: int = 32) -> dict[str, float]:
    max_zone = patch_q.size
    p = np.zeros((levels, max_zone), dtype=np.float64)
    for gray in np.unique(patch_q):
        cc = cc_label(patch_q == gray, connectivity=2)
        if cc.max() == 0:
            continue
        sizes = np.bincount(cc.ravel())[1:]
        for s in sizes:
            p[int(gray), s - 1] += 1
    total = p.sum()
    if total == 0:
        return {k: np.nan for k in ('glszm_sze', 'glszm_lze', 'glszm_gln', 'glszm_zsn')}
    zones = np.arange(1, p.shape[1] + 1)[None, :]
    return {
        'glszm_sze': float((p / (zones * zones)).sum() / total),
        'glszm_lze': float((p * (zones * zones)).sum() / total),
        'glszm_gln': float(((p.sum(axis=1) ** 2).sum()) / total),
        'glszm_zsn': float(((p.sum(axis=0) ** 2).sum()) / total),
    }


def gldm_features_patch(patch_q: np.ndarray, levels: int = 32, delta: int = 1) -> dict[str, float]:
    pad = np.pad(patch_q, 1, mode='edge')
    dep = np.zeros_like(patch_q, dtype=np.int32)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nei = pad[1 + dr:1 + dr + patch_q.shape[0], 1 + dc:1 + dc + patch_q.shape[1]]
            dep += (np.abs(nei.astype(int) - patch_q.astype(int)) <= delta)
    dep += 1
    max_dep = dep.max()
    p = np.zeros((levels, max_dep), dtype=np.float64)
    for g in range(levels):
        m = patch_q == g
        if m.any():
            counts = np.bincount(dep[m], minlength=max_dep + 1)[1:]
            p[g, :len(counts)] += counts
    total = p.sum()
    if total == 0:
        return {k: np.nan for k in ('gldm_sde', 'gldm_lde', 'gldm_gln', 'gldm_dnu')}
    deps = np.arange(1, p.shape[1] + 1)[None, :]
    return {
        'gldm_sde': float((p / (deps * deps)).sum() / total),
        'gldm_lde': float((p * (deps * deps)).sum() / total),
        'gldm_gln': float(((p.sum(axis=1) ** 2).sum()) / total),
        'gldm_dnu': float(((p.sum(axis=0) ** 2).sum()) / total),
    }


def ngtdm_features_patch(patch_q: np.ndarray, levels: int = 32) -> dict[str, float]:
    q = patch_q.astype(np.float32)
    k = 3
    neigh_sum = ndimage.uniform_filter(q, size=k, mode='nearest') * (k * k)
    neigh_mean = (neigh_sum - q) / (k * k - 1)
    s = np.zeros(levels, dtype=np.float64)
    n = np.zeros(levels, dtype=np.float64)
    for g in range(levels):
        m = patch_q == g
        if m.any():
            s[g] = np.abs(q[m] - neigh_mean[m]).sum()
            n[g] = m.sum()
    N = n.sum()
    if N == 0:
        return {k: np.nan for k in ('ngtdm_coarseness', 'ngtdm_contrast', 'ngtdm_busyness', 'ngtdm_complexity', 'ngtdm_strength')}
    p = n / N # proportion at each gray level in n
    gs = np.arange(levels, dtype=np.float64)
    ii, jj = np.meshgrid(gs, gs, indexing='ij')
    pi, pj = np.meshgrid(p, p, indexing='ij')
    diff = np.abs(ii - jj)
    ng = max(int((n > 0).sum()), 1)
    coarseness = 1.0 / max((p * s).sum(), 1e-8) # higher is smoother
    contrast = (pi * pj * diff * diff).sum() * s.sum() / max(ng * (ng - 1) * N, 1e-8)
    busyness = (p * s).sum() / max(np.abs((gs[:, None] * pi) - (gs[None, :] * pj)).sum(), 1e-8)
    complexity = (diff * (pi * s[:, None] + pj * s[None, :]) / (pi + pj + 1e-8)).sum() / max(N, 1e-8)
    strength = ((pi + pj) * diff * diff).sum() / max(s.sum(), 1e-8)
    return {
        'ngtdm_coarseness': float(coarseness),
        'ngtdm_contrast': float(contrast),
        'ngtdm_busyness': float(busyness),
        'ngtdm_complexity': float(complexity),
        'ngtdm_strength': float(strength),
    }

def lbp_features_patch(patch: np.ndarray, levels: int = 32) -> dict[str, float]:
    patch_u = robust_rescale_uint(patch, levels=int(levels))
    lbp = local_binary_pattern(patch_u, P=8, R=1, method='uniform')
    hist, _ = np.histogram(lbp, bins=np.arange(12), density=True)
    return {
        'lbp_mean': float(lbp.mean()),
        'lbp_std': float(lbp.std()),
        'lbp_entropy': float(scipy_entropy(hist + 1e-12, base=2)),
        'lbp_uniformity': float((hist * hist).sum()),
    }


def gradient_orientation_features_patch(patch: np.ndarray) -> dict[str, float]:
    gx = ndimage.sobel(patch, axis=1)
    gy = ndimage.sobel(patch, axis=0)
    mag = np.hypot(gx, gy)
    jxx = ndimage.gaussian_filter(gx * gx, 1.0).mean()
    jyy = ndimage.gaussian_filter(gy * gy, 1.0).mean()
    jxy = ndimage.gaussian_filter(gx * gy, 1.0).mean()
    tr = jxx + jyy
    det = (jxx - jyy) ** 2 + 4 * jxy * jxy
    lam1 = 0.5 * (tr + np.sqrt(max(det, 0.0)))
    lam2 = 0.5 * (tr - np.sqrt(max(det, 0.0)))
    coherence = (lam1 - lam2) / max(lam1 + lam2, 1e-8)
    return {
        'grad_mean': float(mag.mean()),
        'grad_std': float(mag.std()),
        'orientation_coherence': float(coherence),
    }


# ---------- filters ----------

def haar_like_bands(image: np.ndarray) -> dict[str, np.ndarray]:
    """Full-resolution Haar-like bands using separable low/high kernels."""
    img = image.astype(np.float32)
    low = np.array([0.5, 0.5], dtype=np.float32)
    high = np.array([0.5, -0.5], dtype=np.float32)

    def sep(k0, k1):
        out = ndimage.convolve1d(img, k0, axis=0, mode='reflect')
        out = ndimage.convolve1d(out, k1, axis=1, mode='reflect')
        return out

    return {
        'raw': img,
        'haar_ll': sep(low, low),
        'haar_lh': np.abs(sep(low, high)), # the abs gets rid of the directionality and moreso makes an energy map
        'haar_hl': np.abs(sep(high, low)),
        'haar_hh': np.abs(sep(high, high)),
    }


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


# ---------- dense map engine ----------


def _feature_dict_for_patch(
    patch: np.ndarray,
    patch_q: np.ndarray,
    families: tuple[str, ...],
    levels: int,
) -> dict[str, float]:
    out = {}
    if 'firstorder' in families:
        out.update(_nanmean_std_percentiles(patch, bins=levels))
    if 'glcm' in families:
        out.update(glcm_features_patch(patch_q, levels=levels))
    if 'glrlm' in families:
        out.update(glrlm_features_patch(patch_q, levels=levels))
    if 'glszm' in families:
        out.update(glszm_features_patch(patch_q, levels=levels))
    if 'gldm' in families:
        out.update(gldm_features_patch(patch_q, levels=levels))
    if 'ngtdm' in families:
        out.update(ngtdm_features_patch(patch_q, levels=levels))
    if 'lbp' in families:
        out.update(lbp_features_patch(patch, levels=levels))
    if 'gradient' in families:
        out.update(gradient_orientation_features_patch(patch))
    return out

def _row_worker(
    r: int,
    cols: np.ndarray,
    images: dict[str, np.ndarray],
    quantized: dict[str, np.ndarray],
    window: int,
    mask: np.ndarray | None,
    min_valid_frac: float,
    families: tuple[str, ...],
    levels: int,
) -> list[tuple[int, dict[str, float]]]:
    rad = window // 2
    row_results = []
    for c in cols:
        r0, r1 = r - rad, r + rad + 1
        c0, c1 = c - rad, c + rad + 1
        patch_mask = None if mask is None else mask[r0:r1, c0:c1]
        if patch_mask is not None and patch_mask.mean() < min_valid_frac:
            row_results.append((c, {}))
            continue
        feats = {}
        for band_name, image in images.items():
            patch = image[r0:r1, c0:c1]
            patch_q = quantized[band_name][r0:r1, c0:c1]
            patch_feats = _feature_dict_for_patch(patch, patch_q, families=families, levels=levels)
            for k, v in patch_feats.items():
                feats[f'{band_name}__{k}'] = v
        row_results.append((c, feats))
    return row_results


def compute_dense_texture_maps(
    image: np.ndarray,
    window: int = 31,
    step: int = 2,
    mask: np.ndarray | None = None,
    families: tuple[str, ...] = ('firstorder', 'glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm', 'lbp', 'gradient'),
    include_wavelet: bool = True,
    levels: int = 32,
    min_valid_frac: float = 0.5,
    n_jobs: int = 1,
) -> tuple[dict[str, np.ndarray], DenseMapMeta]:
    """
    Generic dense texture engine.

    Output maps live on the sampled grid. Use step=1 or 2 for dense maps, larger step for fast previews.
    """
    img = image.astype(np.float32)
    if img.ndim != 2:
        raise ValueError('compute_dense_texture_maps expects one 2D image')
    if window % 2 == 0:
        raise ValueError('window must be odd')

    bands = haar_like_bands(img) if include_wavelet else {'raw': img}
    quantized = {name: robust_rescale_uint(arr, levels=levels) for name, arr in bands.items()}

    rad = window // 2
    rows = np.arange(rad, img.shape[0] - rad, step)
    cols = np.arange(rad, img.shape[1] - rad, step)
    if len(rows) == 0 or len(cols) == 0:
        raise ValueError('window too large for image')

    worker = delayed(_row_worker)
    results = Parallel(n_jobs=n_jobs, prefer='processes')(
        worker(r, cols, bands, quantized, window, mask, min_valid_frac, families, levels)
        for r in rows
    )

    feature_names = set()
    for row in results:
        for _, feats in row:
            feature_names.update(feats)
    maps = {name: np.full((len(rows), len(cols)), np.nan, dtype=np.float32) for name in sorted(feature_names)}

    for i, row in enumerate(results):
        for j, (_, feats) in enumerate(row):
            for name, value in feats.items():
                maps[name][i, j] = value

    meta = DenseMapMeta(
        row_centers=rows,
        col_centers=cols,
        window=window,
        step=step,
        image_shape=img.shape,
    )
    return maps, meta


def resample_map_to_image(feature_map: np.ndarray, meta: DenseMapMeta, order: int = 1) -> np.ndarray:
    """Resize a sampled-grid map back to image size for quick visualization."""
    return resize(feature_map, meta.image_shape, order=order, preserve_range=True, anti_aliasing=False).astype(np.float32)


def _compute_one_bscan_texture_fullres(
        z: int,
        bscan: np.ndarray,
        upper: np.ndarray,
        lower: np.ndarray,
        step: int,
        pad: int,
        families: tuple[str, ...],
        include_wavelet: bool,
        texture_params: TextureSweepParams,
        features_to_keep: tuple[str, ...] | None = None,
        n_jobs: int = 1,
    ) -> tuple[int, dict[str, np.ndarray]]:
    """
    Compute selected dense texture maps for one B-scan, then resample/scatter them
    back into full (Y, X) image coordinates with NaN outside the processed crop.

    I think really to get the best texture features we should pre-flatten (like for the horizontal and diagnoal runs, the horizontal structures will come out much better)
    """
    texture_params = texture_params.concrete()
    y_min = int(np.floor(np.minimum(upper, lower).min() - pad - texture_params.window))
    y_max = int(np.ceil(np.maximum(upper, lower).max() + pad + texture_params.window))
    y_min = max(0, y_min)
    y_max = min(bscan.shape[0], y_max)


    crop = bscan[y_min:y_max]
    crop = preprocess_texture_image(crop, texture_params)

    maps, meta = compute_dense_texture_maps(
        crop,
        window=int(texture_params.window),
        step=step,
        mask=None,
        families=families,
        include_wavelet=include_wavelet,
        levels=int(texture_params.levels),
        n_jobs=n_jobs,
    )

    # crop = bscan[y_min:y_max]

    # maps, meta = compute_dense_texture_maps(
    #     crop,
    #     window=window,
    #     step=step,
    #     mask=None,
    #     families=families,
    #     include_wavelet=include_wavelet,
    #     levels=levels,
    #     n_jobs=n_jobs,
    # )

    if features_to_keep is not None:
        keep = set(features_to_keep)
        maps = {k: v for k, v in maps.items() if k in keep}

    full_maps = {}
    for name, fmap in maps.items():
        full_crop = resample_map_to_image(fmap, meta)
        full_img = np.full(bscan.shape, np.nan, dtype=np.float32)
        full_img[y_min:y_max, :] = full_crop.astype(np.float32, copy=False)
        full_maps[name] = full_img

    return z, full_maps


def _open_texture_zarr_group(
    out_zarr_path: str | Path,
    volume_shape: tuple[int, int, int],
    feature_names: list[str],
    chunks: tuple[int, int, int] | None = None,
    overwrite: bool = True,
):
    """
    Create one zarr group with one dataset per feature, each shaped (Z, Y, X).
    """
    out_zarr_path = Path(out_zarr_path)
    compressor = numcodecs.Blosc(
        cname='lz4',
        clevel=3,
        shuffle=numcodecs.Blosc.BITSHUFFLE,
    )

    if chunks is None:
        zdim, ydim, xdim = volume_shape
        chunks = (1, min(256, ydim), min(256, xdim))

    # root = zarr.open_group(str(out_zarr_path), mode='w' if overwrite else 'a')
    root = zarr.open_group(
        str(out_zarr_path),
        mode='w' if overwrite else 'a',
        zarr_format=2,
    )
    datasets = {}
    for name in feature_names:
        datasets[name] = root.create_array(
                name=name,
                shape=volume_shape,
                chunks=chunks,
                dtype=np.float32,
                compressor=compressor,
                overwrite=overwrite,
                fill_value=np.nan,
        )
    root.attrs['volume_shape'] = tuple(int(v) for v in volume_shape)
    root.attrs['chunks'] = tuple(int(v) for v in chunks)
    return root, datasets


def compute_bscan_texture_volumes_to_zarr(
    volume: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    out_zarr_path: str | Path,
    z_step: int = 1,
    window: int = 31,
    step: int = 4,
    pad: int = 10,
    families: tuple[str, ...] = ('firstorder', 'glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm', 'lbp', 'gradient'),
    include_wavelet: bool = False,
    levels: int = 32,
    texture_params: TextureSweepParams | None = None,
    features_to_keep: tuple[str, ...] | None = None,
    chunks: tuple[int, int, int] | None = None,
    n_jobs: int = 1,
    single_bscan_n_jobs: int = 1,
    overwrite: bool = True,
) -> Path:
    """
    Compute dense B-scan texture maps and stream them into a zarr group.

    Output layout:
        out_zarr_path/
            raw__mean
            raw__std
            raw__glcm_contrast
            ...

    Each dataset has shape (Z, Y, X), with NaN for unsampled z slices when z_step > 1.
    """

    #for backwards compat
    if texture_params is None:
        texture_params = TextureSweepParams(
            window=window,
            levels=levels,
            gaussian_sigma=0.0,
            downsample_factor=1,
        )
    texture_params = texture_params.concrete()
    
    vol = np.asarray(volume)
    if vol.ndim != 3:
        raise ValueError('volume must be (Z, Y, X)')
    if upper_bound.shape != (vol.shape[0], vol.shape[2]):
        raise ValueError('upper_bound must have shape (Z, X)')
    if lower_bound.shape != (vol.shape[0], vol.shape[2]):
        raise ValueError('lower_bound must have shape (Z, X)')

    z_idx = np.arange(0, vol.shape[0], z_step, dtype=int)
    if len(z_idx) == 0:
        raise ValueError('No z indices selected')

    # REFACTOR can make faster, notable for small slice count
    # Probe first slice to determine feature names before opening zarr datasets.
    z0 = int(z_idx[0])
    _, first_maps = _compute_one_bscan_texture_fullres(
        z=z0,
        bscan=vol[z0].astype(np.float32),
        upper=upper_bound[z0].astype(np.float32),
        lower=lower_bound[z0].astype(np.float32),
        step=step,
        pad=pad,
        families=families,
        include_wavelet=include_wavelet,
        texture_params=texture_params,
        features_to_keep=features_to_keep,
        n_jobs=single_bscan_n_jobs,
    )
    feature_names = sorted(first_maps)
    if not feature_names:
        raise ValueError('No feature maps were produced')

    root, zarr_datasets = _open_texture_zarr_group(
        out_zarr_path=out_zarr_path,
        volume_shape=vol.shape,
        feature_names=feature_names,
        chunks=chunks,
        overwrite=overwrite,
    )

    root.attrs['z_step'] = int(z_step)
    root.attrs['window'] = int(texture_params.window)
    root.attrs['step'] = int(step)
    root.attrs['pad'] = int(pad)
    root.attrs['levels'] = int(texture_params.levels)
    root.attrs['gaussian_sigma'] = float(texture_params.gaussian_sigma)
    root.attrs['downsample_factor'] = int(texture_params.downsample_factor)
    root.attrs['families'] = list(families)
    root.attrs['include_wavelet'] = bool(include_wavelet)
    root.attrs['features_to_keep'] = None if features_to_keep is None else list(features_to_keep)
    root.attrs['texture_params'] = texture_params.as_attrs()

    # Write the probe slice immediately.
    for name, arr in first_maps.items():
        zarr_datasets[name][z0, :, :] = arr

    remaining_z = [int(z) for z in z_idx[1:]]

    if n_jobs == 1:
        for z in remaining_z:
            _, maps = _compute_one_bscan_texture_fullres(
                z=z,
                bscan=vol[z].astype(np.float32),
                upper=upper_bound[z].astype(np.float32),
                lower=lower_bound[z].astype(np.float32),
                step=step,
                pad=pad,
                families=families,
                include_wavelet=include_wavelet,
                texture_params=texture_params,
                features_to_keep=features_to_keep,
                n_jobs=single_bscan_n_jobs,
            )
            for name, arr in maps.items():
                zarr_datasets[name][z, :, :] = arr
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futures = [
                ex.submit(
                    _compute_one_bscan_texture_fullres,
                    int(z),
                    vol[z].astype(np.float32),
                    upper_bound[z].astype(np.float32),
                    lower_bound[z].astype(np.float32),
                    step,
                    pad,
                    families,
                    include_wavelet,
                    texture_params,
                    features_to_keep,
                    single_bscan_n_jobs,
                )
                for z in remaining_z
            ]

            for fut in as_completed(futures):
                z, maps = fut.result()
                for name, arr in maps.items():
                    zarr_datasets[name][z, :, :] = arr

    zarr.consolidate_metadata(str(out_zarr_path))
    return Path(out_zarr_path)


# ---------- B-scan -> en-face ----------

def _project_one_bscan(
    z: int,
    bscan: np.ndarray,
    upper: np.ndarray,
    lower: np.ndarray,
    window: int,
    step: int,
    pad: int,
    families: tuple[str, ...],
    include_wavelet: bool,
    levels: int,
    feature_post_radius: int,
    debug_plot: bool=False,
) -> tuple[int, dict[str, np.ndarray]]:
    y_min = int(np.floor(np.minimum(upper, lower).min() - pad - window))
    y_max = int(np.ceil(np.maximum(upper, lower).max() + pad + window))
    y_min = max(0, y_min)
    y_max = min(bscan.shape[0], y_max)
    crop = bscan[y_min:y_max]

    maps, meta = compute_dense_texture_maps(
        crop,
        window=window,
        step=step,
        mask=None,
        families=families,
        include_wavelet=include_wavelet,
        levels=levels,
        n_jobs=1,
    )

    # AB = spu.ArrayBoard(skip = -debug_plot,plt_display=False,save_tag=f"bscan_demo")
    # for map in maps:
        # do the array board stuff here


    # Nonsense!
    # shadow_mask = estimate_shadow_mask_from_bscan(crop)
    # shadow_grid = shadow_mask[np.ix_(meta.row_centers, meta.col_centers)]
    # maps = postprocess_feature_dict(maps, vessel_mask=shadow_grid, radius=feature_post_radius)

    y_grid = meta.row_centers + y_min
    x_grid = meta.col_centers
    vecs = {}
    for name, fmap in maps.items():
        sampled = np.full(len(x_grid), np.nan, dtype=np.float32)
        for j, x in enumerate(x_grid):
            lo = min(upper[x], lower[x]) - pad
            hi = max(upper[x], lower[x]) + pad
            keep = (y_grid >= lo) & (y_grid <= hi)
            if keep.any():
                sampled[j] = np.nanmean(fmap[keep, j]) # This is very opnioninaed taking the mean only
        full = np.full(bscan.shape[1], np.nan, dtype=np.float32)
        good = np.isfinite(sampled)
        if good.sum() >= 2:
            full[:] = np.interp(np.arange(bscan.shape[1]), x_grid[good], sampled[good])
        vecs[name] = full
    return z, vecs


def project_bscan_texture_to_enface(
    volume: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    z_step: int = 1,
    window: int = 31,
    step: int = 4,
    pad: int = 10,
    families: tuple[str, ...] = ('firstorder', 'glcm', 'glrlm', 'glszm', 'gldm', 'ngtdm', 'lbp', 'gradient'),
    include_wavelet: bool = False,
    levels: int = 32,
    feature_post_radius: int = 3,
    n_jobs: int = 1,
) -> dict[str, np.ndarray]:
    """
    likely not going to be used -- we will instead have the bscan texture zarrs and use some of the below functions.
    Compute texture in each B-scan, then collapse vertically inside the ROI to build en-face feature maps.

    upper_bound and lower_bound should be (Z, X) arrays.
    """
    vol = np.asarray(volume)
    if vol.ndim != 3:
        raise ValueError('volume must be (Z, Y, X)')
    z_idx = np.arange(0, vol.shape[0], z_step)
    worker = delayed(_project_one_bscan)
    out = Parallel(n_jobs=n_jobs, prefer='processes')(
        worker(
            int(z),
            vol[z].astype(np.float32),
            upper_bound[z].astype(np.float32),
            lower_bound[z].astype(np.float32),
            window,
            step,
            pad,
            families,
            include_wavelet,
            levels,
            feature_post_radius,
        )
        for z in z_idx
    )

    feature_names = sorted({k for _, d in out for k in d})
    enface = {name: np.full((vol.shape[0], vol.shape[2]), np.nan, dtype=np.float32) for name in feature_names}
    for z, vecs in out:
        for name, vec in vecs.items():
            enface[name][z] = vec

    if z_step > 1:
        filled = {}
        src_z = z_idx.astype(float)
        full_z = np.arange(vol.shape[0], dtype=float)
        for name, arr in enface.items():
            out_arr = arr.copy()
            for x in range(arr.shape[1]):
                good = np.isfinite(arr[z_idx, x])
                if good.sum() >= 2:
                    out_arr[:, x] = np.interp(full_z, src_z[good], arr[z_idx[good], x])
            filled[name] = out_arr
        enface = filled
    return enface

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

def project_texture_zarr_to_enface_for_volume(
    vol_path,
    texture_zarr_path,
    flat_layers_npz,
    line_key=None,
    candidate_slabs = [[5,15]],
    features=None,
    interp_x=True,
):
    """
    Load a thick texture zarr and re-project a thinner slab to en-face.

    candidate_slabs are [bottom_idx offset from RPE line, top index offset from RPE line (lower y value)]
    line_key:
      None -> use file_utils.get_algorithm_key_from_filepath(vol_path)
      str  -> use that flattened layer key directly
    """
    from pathlib import Path
    import numpy as np
    import zarr
    from code_files import file_utils

    vol_path = Path(vol_path)
    flat_layers = np.load(flat_layers_npz)

    if line_key is None:
        line_key = file_utils.get_algorithm_key_from_filepath(vol_path)

    root = zarr.open_group(str(texture_zarr_path), mode='r')
    center = flat_layers[line_key]

    if features is None:
        features = list(root.array_keys())

    out = {}
    for feat in features:
        proj = TextureSlabProjector(
                    texture_vol=root[feat],   # pass zarr-backed array, not full np.asarray
                    reference_line=center,
                    max_top_offset=min(t for b, t in candidate_slabs),
                    max_bottom_offset=max(b for b, t in candidate_slabs),
                    extra_pad=2,
                )
        for bottom_offset, top_offset in candidate_slabs:
            enface = proj.project_between(center - top_offset, center - bottom_offset,interp_x=interp_x)
            out[f"{feat}|{bottom_offset}->{top_offset}"] = enface
    return out


def slab_average_to_enface(
    volume: np.ndarray,
    upper_bound: np.ndarray,
    lower_bound: np.ndarray,
    pad: int = 5,
    statistic: str = 'mean',
) -> np.ndarray:
    """Collapse each B-scan slab directly into an en-face image before any texture computation."""
    stat_fn = np.nanmean if statistic == 'mean' else np.nanmedian
    vol = np.asarray(volume)
    out = np.full((vol.shape[0], vol.shape[2]), np.nan, dtype=np.float32)
    for z in range(vol.shape[0]):
        for x in range(vol.shape[2]):
            lo = max(0, int(np.floor(min(upper_bound[z, x], lower_bound[z, x]) - pad)))
            hi = min(vol.shape[1], int(np.ceil(max(upper_bound[z, x], lower_bound[z, x]) + pad)) + 1)
            if hi > lo:
                out[z, x] = float(stat_fn(vol[z, lo:hi, x]))
    return out

def save_enface_feature_maps_npz(
    feature_maps: dict[str, np.ndarray],
    out_npz_path: str | Path,
) -> Path:
    """
    Save small en-face feature maps, shape (Z, X), into one NPZ.
    """
    out_npz_path = Path(out_npz_path)
    out_npz_path.parent.mkdir(parents=True, exist_ok=True)

    arrays = {
        name: np.asarray(arr, dtype=np.float32)
        for name, arr in feature_maps.items()
    }
    np.savez_compressed(out_npz_path, **arrays)
    return out_npz_path

