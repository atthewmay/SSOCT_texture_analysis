import numpy as np

from scipy.ndimage import convolve, maximum_filter1d
from scipy.signal import savgol_filter
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # adds Han_AIR/ to path
from visualization_utils import PlotTracer
p = PlotTracer(show=False)
p.tracing=True
from collections import Counter
import code_files.segmentation_code.segmentation_plot_utils as spu
import code_files.segmentation_code.segmentation_step_functions as ssf

from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.ndimage import median_filter

import cv2


def _axial_gradient(img, kernel):
    return convolve(img.astype(np.float32), kernel[:, None], mode='reflect')


def _boundary_enhance(img, vertical_kernel_size = 4, dark2bright=True, blur_ksize=40):
    k_up = np.array(np.concatenate([np.repeat(-1,vertical_kernel_size//2),np.repeat(1,vertical_kernel_size//2)]), dtype=np.float32)
    g = _axial_gradient(img, k_up if dark2bright else -k_up)
    g[g < 0] = 0
    g /= (g.max() + 1e-6) # normalize
    ctx = cv2.blur(img.astype(np.float32), (blur_ksize,blur_ksize))
    ctx /= (ctx.max() + 1e-6)

    enh = g * ctx
    return enh / (enh.max() + 1e-6)

def _normalized_axial_gradient(img,vertical_kernel_size,dark2bright):
    """not DRY"""
    k_up = np.array(np.concatenate([np.repeat(-1,vertical_kernel_size//2),np.repeat(1,vertical_kernel_size//2)]), dtype=np.float32)
    g = _axial_gradient(img, k_up if dark2bright else -k_up)
    g[g < 0] = 0
    g /= (g.max() + 1e-6) # normalize
    return g

def _blur_image(img,blur_k_x,blur_k_y,blur_type='blur'):
    """allows anisotropic blur"""
    if blur_type=='blur':
        blurred_img = cv2.blur(img.astype(np.float32), (blur_k_x,blur_k_y))
        blurred_img /= (blurred_img.max() + 1e-6)
    elif blur_type == 'median':
        blurred_img = median_filter(img, size=(blur_k_y, blur_k_x))  # 1 row tall, 51 cols wide
    return blurred_img



def _nms_columnwise(enh, radius=4, thresh=0.08, *, vertical_filter=None, value_filter=None,keeptop=True,narrow_radius_loop=False):
    """
    Find column-wise local maxima above a threshold, then optionally
    keep only the top-k by enhancement value or by vertical position.

    Parameters
    ----------
    enh : 2D array
        Enhanced image.
    radius : int
        Neighborhood radius for non-maximum suppression.
    thresh : float
        Minimum enhancement value for a maxima to be considered.
    vertical_filter : int or None
        If not None, keep only this many maxima per column with the
        smallest row indices (i.e. highest in the image).
    value_filter : int or None
        If not None, keep only this many maxima per column with the
        largest enhancement values.

        keeptop is to say when using vertical filter, keep the topmost vs bottom

    Returns

    concern: hmm. Well I'm worried for the two-peak situation that if one peak is accidentally flat-topped this will fail.
    This coudl be an area for improvement. 
    -------
    maxima : 2D bool array
    """
    H, W = enh.shape

    # basic NMS + threshold
    maxima = enh == maximum_filter1d(enh, size=radius, axis=0, mode='reflect')
    maxima &= (enh > thresh)

    if narrow_radius_loop:
        while len(maxima)<2:
            radius /= 2
            maxima = enh == maximum_filter1d(enh, size=radius, axis=0, mode='reflect')


    # for each column, optionally filter by enhancement value
    if value_filter is not None:
        for c in range(W):
            rows = np.nonzero(maxima[:, c])[0]
            if len(rows) > value_filter:
                # sort rows by descending enhancement value
                order = np.argsort(enh[rows, c])[::-1]
                keep = set(rows[order[:value_filter]])
                mask = np.zeros_like(maxima[:, c], dtype=bool)
                mask[list(keep)] = True
                maxima[:, c] &= mask

    # for each column, optionally filter by vertical position (top of image)
    if vertical_filter is not None:
        for c in range(W):
            rows = np.nonzero(maxima[:, c])[0]
            if len(rows) > vertical_filter:
                # keep the smallest row indices (highest pixels)
                if keeptop==True:
                    keep = set(np.sort(rows)[:vertical_filter])
                else:
                    keep = set(np.sort(rows)[-vertical_filter:])
                mask = np.zeros_like(maxima[:, c], dtype=bool)
                mask[list(keep)] = True
                maxima[:, c] &= mask

    return maxima

def peaks_to_seeds(peaks_by_col, H, W=None):
    """
    peaks_by_col: list length W, each entry is an iterable of peak y-indices for that column.
                  e.g. peaks_by_col[c] = [12, 40, 41]
    Returns:
      seeds: (H,W) bool, True at (y,c) for each peak.
    """
    if W is None:
        W = len(peaks_by_col)

    seeds = np.zeros((H, W), dtype=bool)
    for c, ys in enumerate(peaks_by_col[:W]):
        ys = np.asarray(list(ys), dtype=int)
        seeds[ys, c] = True
    return seeds

def line_to_seeds(line, H):
    ys = np.asarray(line, dtype=float)
    W = ys.size
    xs = np.arange(W, dtype=int)

    ok = np.isfinite(ys)
    ys_i = ys[ok].astype(int)
    ys_i = np.clip(ys_i, 0, H - 1)

    seeds = np.zeros((H, W), dtype=bool)
    seeds[ys_i, xs[ok]] = True
    return seeds



#parallel version

from collections import Counter
import numpy as np

def _trace_one_seed(args):
    enh, offsets, break_zeros, r0, c0 = args
    H, W = enh.shape
    paths = []
    counter = Counter()

    for dc in (+1, -1):
        path = [(r0, c0)]
        r, c = r0, c0
        contig_breaker = 0

        while True:
            if not (0 <= c + dc < W):
                break

            cand_r = r + offsets
            cand_r = cand_r[(cand_r >= 0) & (cand_r < H)]

            diffs = np.abs(enh[cand_r, c + dc] - enh[r, c])

            if np.ptp(diffs) == 0:
                if break_zeros:
                    if np.all(diffs == 0):
                        contig_breaker += 1
                        if contig_breaker > 10:
                            counter.update(["path_broken"])
                            break
                    else:
                        contig_breaker = 0
                r_new = int(cand_r[len(cand_r) // 2])
                counter.update(["tie"])
            else:
                r_new = int(cand_r[int(diffs.argmin())])
                counter.update(["yes_difference"])

            r, c = r_new, c + dc
            path.append((r, c))

        paths.append(path)

    return paths, counter


def _trace_paths(enh, seeds, neighbourhood=1, break_zeros=False, nworkers=1):
    H, W = enh.shape
    offsets = np.arange(-neighbourhood, neighbourhood + 1)

    rs, cs = np.nonzero(seeds)
    jobs = [(enh, offsets, break_zeros, int(r0), int(c0)) for r0, c0 in zip(rs, cs)]

    paths = []
    counter = Counter()

    if nworkers <= 1:
        for j in jobs:
            p, c = _trace_one_seed(j)
            paths.extend(p)
            counter.update(c)
        return paths  # (or return paths, counter if you want)

    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=nworkers) as ex:
        for p, c in ex.map(_trace_one_seed, jobs, chunksize=64):
            paths.extend(p)
            counter.update(c)

    return paths



# def _trace_paths(enh, seeds, neighbourhood=1,break_zeros=False):
#     H, W = enh.shape
#     offsets = np.arange(-neighbourhood, neighbourhood + 1)
#     paths = []
#     counter = Counter()
#     for r0, c0 in zip(*np.nonzero(seeds)):

#         # added: trace in both directions from each seed
#         for dc in (+1, -1):
#             path = [(r0, c0)]
#             r, c = r0, c0
#             contig_breaker=0
#             while True:
#                 # guard next column
#                 if not (0 <= c + dc < W):
#                     break

#                 # find candidate rows
#                 cand_r = r + offsets
#                 cand_r = cand_r[(cand_r >= 0) & (cand_r < H)]

#                 # choose next row by looking *ahead* at c+dc
#                 diffs = np.abs(enh[cand_r, c + dc] - enh[r, c])
                    

#                 # fast check: are all entries in diffs identical?
#                 # (ptp = max–min)
#                 if np.ptp(diffs) == 0:
#                     # tie: pick the median candidate
#                     if break_zeros:
#                         if np.all(diffs==0):
#                             contig_breaker += 1 
#                             if contig_breaker > 10:
#                                 counter.update(['path_broken'])
#                                 break
#                         else:
#                             contig_breaker=0
#                     r_new = int(cand_r[len(cand_r)//2])
#                     counter.update(diffs)
#                 else:
#                     # normal case: choose the minimum-difference row
#                     idx = diffs.argmin()
#                     r_new = int(cand_r[idx])
#                     counter.update(["yes_difference"])

#                 # advance
#                 r, c = r_new, c + dc
#                 path.append((r, c))

#             paths.append(path)
#     return paths


def _probability_image(paths, shape):
    prob = np.zeros(shape, dtype=np.float32)
    for path in paths:
        for r, c in path:
            prob[r, c] += 1
    return prob / (prob.max() + 1e-6)


def _hysteresis(prob, high=0.15, low=0.02):
    strong = prob >= high
    weak = (prob >= low) & ~strong
    mask = strong.copy()
    changed = True
    while changed:
        changed = False
        dilated = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        add = dilated & weak & ~mask
        if add.any():
            mask |= add
            changed = True
    return mask


def _extract_topbottom_line(edge_img, skip_extreme=5,direction='bottom'):
    H, W = edge_img.shape
    rows = np.full(W, np.nan)
    for c in range(W):
        rs = np.where(edge_img[:, c])[0]
        if direction == 'bottom':
            # rs = rs[rs < H - skip_extreme]
            if len(rs)>skip_extreme:
                rs = rs if skip_extreme==0 else rs[:-skip_extreme]
            if rs.size:
                rows[c] = rs.max()
        elif direction == 'top':
            if len(rs)>skip_extreme:
                rs = rs[skip_extreme:]
            # rs = rs[rs >= skip_extreme]
            if rs.size:
                rows[c] = rs.min()
    good = ~np.isnan(rows)
    return np.interp(np.arange(W), np.flatnonzero(good), rows[good]) if good.sum() > 1 else rows


def smooth_rpe_line(rpe_raw, rigidity=61, polyorder=3):
    W = len(rpe_raw)
    win = min(rigidity, W - (1 - W % 2))
    if win <= polyorder:
        win = polyorder + 3
    if win % 2 == 0:
        win += 1
    return savgol_filter(rpe_raw, int(win), int(polyorder), mode='interp')



def downward_horizontal_blur(img: np.ndarray,
                             k_rows: int,
                             k_cols: int) -> np.ndarray:
    """
    One-sided vertical + symmetric horizontal mean blur.

    Parameters
    ----------
    img : (H, W) array
        Input image.
    k_rows : int
        Number of rows to average *downward* (must be >=1).
    k_cols : int
        Width of horizontal averaging window (must be odd >=1).

    Returns
    -------
    blurred : (H, W) array
        The blurred image.
    """
    H, W = img.shape
    # how much to pad
    pad_vert   = k_rows
    pad_horiz  = k_cols // 2

    # pad bottom and left/right by edge-replication
    padded = np.pad(img,
                    ((0, pad_vert),
                     (pad_horiz, pad_horiz)),
                    mode='edge')            # shape (H+pad_vert, W+2*pad_horiz)

    # vertical cumulative sum
    csum_v = padded.cumsum(axis=0)       # shape (H+pad_vert, W+2*pad_horiz)
    # sum over each downward window of k_rows
    vert_sum = (csum_v[k_rows : k_rows + H]     # rows k_rows ... k_rows+H-1
                - csum_v[0 : H])                # rows 0 ... H-1
    # now vert_sum.shape == (H, W+2*pad_horiz)

    # horizontal cumulative sum
    csum_h = vert_sum.cumsum(axis=1)     # along columns
    # sum over each centered window of k_cols
    horiz_sum = (csum_h[:, k_cols : k_cols + W]  # cols k_cols ... k_cols+W-1
                 - csum_h[:, 0 : W])             # cols 0 ... W-1
    # now horiz_sum.shape == (H, W)

    # normalize by total kernel area
    horiz_sum /= (horiz_sum.max())
    return horiz_sum 

def compute_enh_diff(img, blur_ksize, vks_false,vks_true,k_rows,k_cols):
    """
    Returns enh_false, enh_true, diff_norm as before.
    """
    enh_t = _boundary_enhance(
        img,
        vertical_kernel_size=vks_true,
        dark2bright=True,
        blur_ksize=blur_ksize
    )
    enh_f = _boundary_enhance(
        img,
        vertical_kernel_size=vks_false,
        dark2bright=False,
        blur_ksize=blur_ksize
    )
    dwn = downward_horizontal_blur(enh_t,k_rows=k_rows,k_cols=k_cols)
    diff = enh_f - dwn
    diff[diff < 0] = 0
    diff /= (diff.max() + 1e-6)
    return {
        "enh_f":enh_f,
        "enh_t":enh_t,
        "dwn":dwn,
        "diff":diff,
        }

def ILM_impute_missing_edges(edge, enh,total_seeds):
    """
    For columns with no edge (edge.any(axis=0)==False),
    pick the topmost index among the two highest values in enh[:, j]
    and set that pixel True in the edge map.
    """
    edge = edge.astype(bool).copy()
    H, W = enh.shape
    no_edge_cols = ~edge.any(axis=0)

    total_seeds = 2
    seeds = _nms_columnwise(enh, radius=25, thresh=0.08, value_filter=total_seeds)
    for j in np.flatnonzero(no_edge_cols):
        col = seeds[:, j]
        if not np.isfinite(col).any():
            continue
        # get indices of two largest values (or one if H==1)
        if H >= 2:
            idx2 = np.argpartition(col, -total_seeds)[-total_seeds:]
        else:
            idx2 = np.array([int(np.nanargmax(col))])
        r = int(np.min(idx2))  # "top" of the two maxima
        edge[r, j] = True

    return edge


# -------------- Objets from the old segmentation utils file0
class peakSuppressor(object):
    """a grouping of static methods"""

    @staticmethod
    def extract_smoothed_and_peaks(
        bscan: np.ndarray,
        sigma: float = 2.0,
        peak_prominence: float = 0.02,
        peak_distance: int = 5,
        ilm_line: np.ndarray | None = None,  # shape (W,), rows of ILM per column
        min_offset: int = -15,                 # start peak search this many rows below ILM
    ):
        """
        bscan: (H, W). If ilm_line is provided, peaks are searched only at rows >= ilm+min_offset.
        Returns:
        smoothed: (H, W)
        peaks: list of np.ndarray of peak-row indices per column
        topmost: (W,) row of the shallowest found peak per column (nan if none)
        """
        bscan = np.asarray(bscan, dtype=float)
        if bscan.max() > 0:
            bscan = bscan / bscan.max()

        smoothed = gaussian_filter1d(bscan, sigma=sigma, axis=0, mode="nearest")

        H, W = smoothed.shape
        peaks = []
        topmost = np.full(W, np.nan, dtype=float)

        have_ilm = ilm_line is not None
        if have_ilm:
            ilm_line = np.asarray(ilm_line).astype(float)
            if ilm_line.shape != (W,):
                raise ValueError(f"ilm_line must have shape ({W},), got {ilm_line.shape}")

        for c in range(W):
            s = smoothed[:, c]

            # restrict search to inferior to ILM (if provided)
            if have_ilm and np.isfinite(ilm_line[c]):
                start = int(np.floor(ilm_line[c])) + min_offset
                start = max(0, min(start, H))  # clamp to [0, H]
            else:
                start = 0

            if start >= H:
                p_abs = np.array([], dtype=int)
            else:
                p_rel, _ = find_peaks(s[start:], prominence=peak_prominence, distance=peak_distance)
                p_abs = p_rel + start

            peaks.append(p_abs)
            if p_abs.size > 0:
                topmost[c] = p_abs.min()

        return smoothed, peaks, topmost


    @staticmethod
    def align_by_topmost_peak(smoothed: np.ndarray, peaks: list, topmost: np.ndarray):
        """
        Shifts each column so its topmost peak sits at a common reference row.
        Returns:
        aligned: 2D array (H + pad, W) with columns shifted;
        peak_rows_aligned: list of arrays with peak rows after alignment.
        """
        H, W = smoothed.shape
        valid = ~np.isnan(topmost)
        if not np.any(valid):
            raise Exception
            # no peaks anywhere; return original
            # return smoothed.copy(), [np.array([], dtype=int) for _ in range(W)]

        ref = int(np.nanmedian(topmost))  # stable center
        shifts = np.zeros(W, dtype=int)
        shifts[valid] = (topmost[valid].astype(int) - ref)

        up = max(0, shifts.max())
        down = max(0, (-shifts).max())
        out_H = H + up + down

        aligned = np.zeros((out_H, W), dtype=smoothed.dtype)
        peak_rows_aligned = []

        for c in range(W):
            shift = shifts[c]
            dest_start = up - shift
            dest_end = dest_start + H
            aligned[dest_start:dest_end, c] = smoothed[:, c]

            # shift peak rows for this column
            p = peaks[c]
            if p.size:
                peak_rows_aligned.append(p + (up - shift))
            else:
                peak_rows_aligned.append(np.array([], dtype=int))

        return aligned, peak_rows_aligned

    @staticmethod
    def suppress_below_third_peak(
        img_to_alter: np.ndarray,
        peak_source_image: np.ndarray,
        peaks: list,
        margin: int,
        factor: float = 0.5,
        # inplace: bool = False,
    ) -> np.ndarray:
        """
        img: (H, W) image to attenuate (same shape as used for peak detection)
        peaks: list of 1D arrays of peak-row indices per column
        margin: rows above the 3rd peak to start attenuation
        factor: multiplicative attenuation, 0<factor<=1

        Unfortunately this currently doesn't let you input a totally different image and compare the heights of the peaks. Would add later if needed. j
        """
        img_to_alter = np.asarray(img_to_alter, dtype=float).copy()
        # H, W = x.shape
        assert img_to_alter.shape == peak_source_image.shape
        H, W = img_to_alter.shape
        factor = float(np.clip(factor, 0.0, 1.0))
        mrg = int(margin)

        for c in range(W):
            p = np.asarray(peaks[c], dtype=int)
            if p.size < 3:
                continue
            p = np.sort(p)[:3]          # shallowest three peaks
            m = peak_source_image[p, c]                 # their magnitudes in this image
            if m[2] < m[0] and m[2] < m[1]:
                cut = max(0, p[2] - mrg)
                img_to_alter[cut:, c] *= factor

        return img_to_alter

    @staticmethod
    def suppress_below_third_peak_valley(
        img: np.ndarray,
        peaks: list,
        factor: float = 0.5,
        valley_prominence: float = 0.0,
        inplace: bool = False,
    ) -> np.ndarray:
        """
        For each column:
        - take the 3 shallowest peaks (by depth)
        - if the 3rd is lower in magnitude than the 1st and 2nd
        - find a valley between peak #2 and peak #3
        - attenuate everything at/below that valley by `factor`.

        img: (H, W) image whose intensities you want to suppress.
        peaks: list of 1D arrays of peak row indices per column.
        factor: multiplicative attenuation (0<factor<=1).
        valley_prominence: passed to find_peaks on the inverted signal; 0 = any local min.
        """
        x = img if inplace else np.asarray(img, dtype=float).copy()
        H, W = x.shape
        factor = float(np.clip(factor, 0.0, 1.0))

        for c in range(W):
            p = np.asarray(peaks[c], dtype=int)
            if p.size < 3:
                continue

            # first three peaks in depth order (shallowest→deepest)
            p3 = np.sort(p)[:3]
            # print(p3)
            m = x[p3, c]

            # only suppress if 3rd peak is lower than the first two
            if not (m[2] < m[0] and m[2] < m[1]):
                continue

            # search interval between peak #2 and peak #3 (inclusive)
            start = p3[1]
            end = p3[2]
            if end <= start:
                continue

            segment = x[start:end + 1, c]        # [start, end]
            inv_segment = -segment

            # valleys = peaks of the inverted signal
            valley_rel, props = find_peaks(inv_segment, prominence=valley_prominence)

            if valley_rel.size > 0:
                # pick the valley with highest prominence (deepest, cleanest valley)
                idx = np.argmax(props["prominences"])
                valley_row = start + valley_rel[idx]
            else:
                # fallback: global minimum in that region
                valley_row = start + int(np.argmin(segment))

            # suppress at/below valley
            valley_row = int(np.clip(valley_row, 0, H - 1))
            # print("valley row is {valley_row}")
            x[valley_row:, c] *= factor

        return x

    @staticmethod
    def suppress_above_bottom_peak_valley(
        img: np.ndarray,
        peaks: list,
        factor: float = 0,
        valley_prominence: float = 0.0,
        inplace: bool = False,
    ) -> np.ndarray:
        """
        For each column:
        - take the 2 tallest allowed peaks
        - if the 3rd is lower in magnitude than the 1st and 2nd
        - find a valley between peak #2 and peak #3
        - attenuate everything at/below that valley by `factor`.

        img: (H, W) image whose intensities you want to suppress.
        peaks: list of 1D arrays of peak row indices per column.
        factor: multiplicative attenuation (0<factor<=1).
        valley_prominence: passed to find_peaks on the inverted signal; 0 = any local min.
        """
        x = img if inplace else np.asarray(img, dtype=float).copy()
        H, W = x.shape
        factor = float(np.clip(factor, 0.0, 1.0))

        for c in range(W):
            p = np.asarray(peaks[c], dtype=int)
            if p.size < 2:
                continue

            # first three peaks in depth order (shallowest→deepest)
            p = np.sort(p)
            biggest2peaks_y = p[np.argsort(x[p, c])[-2:]] 
            biggest2peaks_y = np.sort(biggest2peaks_y)

            # search interval between peak #2 and peak #3 (inclusive)
            start = biggest2peaks_y[0]
            end = biggest2peaks_y[1]
            # print(p)
            # print(biggest2peaks_y)
            if end <= start:
                raise Exception

            segment = x[start:end + 1, c]        # [start, end]
            inv_segment = -segment

            # valleys = peaks of the inverted signal
            valley_rel, props = find_peaks(inv_segment, prominence=valley_prominence)

            # if valley_rel.size > 0:
            #     # pick the valley with highest prominence (deepest, cleanest valley)
            #     idx = np.argmax(props["prominences"])
            #     valley_row = start + valley_rel[idx]
            # else:
            #     # fallback: global minimum in that region
            valley_row = start + int(np.argmin(segment))

            valley_row = int(np.clip(valley_row, 0, H - 1))
            #suppress above valley_row
            x[:valley_row, c] *= factor

        return x


    @staticmethod
    def suppress_above_below_line(img, line, factor=0.0, px_margin_to_start_suppressing=2, direction="above"):
        img = img.copy()
        H, W = img.shape

        yline = np.asarray(line, dtype=np.float32)
        yline = np.where(np.isnan(yline), -1e9, yline)  # NaN -> suppress nothing (handled below)

        yy = np.arange(H)[:, None]          # (H,1)
        y0 = yline[None, :]                 # (1,W)

        if direction == "above":
            mask = yy < (y0 - px_margin_to_start_suppressing)
        elif direction == "below":
            mask = yy > (y0 + px_margin_to_start_suppressing)
        else:
            raise ValueError("direction must be 'above' or 'below'")

        # if line is NaN at a column, disable mask for that column
        mask &= np.isfinite(y0)

        img[mask] *= factor
        return img




    @staticmethod
    def peak_suppression_pipeline(peak_source_image,image_to_alter,ilm_line,**kwargs):
        """peak source img and img to alter are likely the same images.
        This pipeline works for the enh type gradient images, where you likely have a bright ILM, and bright RPE, with likely some choroidal junk below
        For EZ-vs-RPE distinction, use the EZ_RPE_peak_suppression_pipeline
        """
            # bscan: (H, W) grayscale OCT B-scan
        cfg = dict( sigma=2.0, peak_prominence=0.02, peak_distance=20, min_offset=-15, third_peak_margin=20, suppression_factor=0.1) # Some defaults that get overwritten by kwargs
        cfg.update(kwargs)
        _, peaks, _ = peakSuppressor.extract_smoothed_and_peaks(
                                peak_source_image,
                                sigma=cfg['sigma'],
                                peak_prominence=cfg['peak_prominence'],   # tune per dataset
                                peak_distance=cfg['peak_distance'],
                                ilm_line=ilm_line,
                                min_offset=cfg['min_offset'],
        )

        suppressed_img = peakSuppressor.suppress_below_third_peak_valley(image_to_alter,peaks,
                                                                    factor=cfg['suppression_factor'])

        return suppressed_img


    @staticmethod
    def EZ_RPE_peak_suppression_pipeline(peak_source_image,image_to_alter,ilm_line,**kwargs):
            # bscan: (H, W) grayscale OCT B-scan

        # def add-peak
        cfg = dict( sigma=2.0, peak_prominence=0.2, peak_distance=20, min_offset=-15, suppression_factor=0.1) # Some defaults that get overwritten by kwargs
        cfg.update(kwargs)
        AB = spu.ArrayBoard(plt_display=False,dpi=400,save_tag="_peak_suppress",skip=True)
        AB.add(peak_source_image,title = 'source')
        smoothed, peaks, _ = peakSuppressor.extract_smoothed_and_peaks(
                                peak_source_image,
                                sigma=cfg['sigma'],
                                peak_prominence=cfg['peak_prominence'],   # tune per dataset
                                peak_distance=cfg['peak_distance'],
                                ilm_line=ilm_line,
                                min_offset=cfg['min_offset'],
        )
        # AB.add(smoothed,title='smoothed')
        # # c = smoothed.shape[-1]//2
        # AB.add_plot(lambda ax: peakSuppressor.plot_ascan(ax,smoothed[:,c],peaks[c]),
        #             title = 'peaks test')

        # AB.add(spu.overlay_peaks_on_image(smoothed,peaks),title='with peaks colored')

        # suppressed_img = peakSuppressor.suppress_above_bottom_peak_valley(image_to_alter,peaks,
        #                                                             factor=cfg['suppression_factor'])
        # AB.add(suppressed_img,title='suppressed_img')
        AB.render()
        return peaks



    # ---------------- plotting helpers ---------------- #

    @staticmethod
    def plot_ascan(ax, signal_1d: np.ndarray, peak_rows: np.ndarray, title: str = None,c='blue'):
        ax.plot(signal_1d,c=c)
        if peak_rows.size:
            ax.scatter(peak_rows, signal_1d[peak_rows], s=20)
        ax.set_xlim(0, signal_1d.shape[0] - 1)
        ax.set_xlabel("Depth (rows)")
        ax.set_ylabel("Intensity")
        if title:
            ax.set_title(title)
        ax.axis('off')


    @staticmethod
    def plot_aligned_stack(ax, aligned: np.ndarray, peak_rows_aligned: list, peak_stride: int = 10):
        """
        Shows all A-scans as an image, already aligned by topmost peak.
        Overlays peaks as dots every 'peak_stride' columns to reduce clutter.
        """
        ax.imshow(aligned, cmap="gray", aspect="auto", origin="upper")
        H, W = aligned.shape

        cols = range(0, W, max(1, peak_stride))
        for c in cols:
            p = peak_rows_aligned[c]
            if p.size:
                ax.scatter(np.full_like(p, c), p, s=6)
        ax.axis('off')


    @staticmethod
    def peak_suppression_debug():
        """Run the example showing the peaks for debugging purposes. """
        import pickle
        import matplotlib.pyplot as plt
        results = pickle.load(open("/Users/matthewhunt/Research/Iowa_Research/Han_AIR/scratch_dir/full_results",'rb'))

        cols_to_show = [50, 200, 350]
        fig, ax = plt.subplots(len(list(results)),len(cols_to_show)+2,figsize=(20, 12), tight_layout=True)
        print(f"expecting {ax.shape} subplots")

        for i,(order_idx,ilm_smooth,enh) in enumerate(results):


            # bscan: (H, W) grayscale OCT B-scan
            smoothed, peaks, topmost = peakSuppressor.extract_smoothed_and_peaks(
                enh,
                sigma=2.0,
                peak_prominence=0.2,   # tune per dataset
                ilm_line=ilm_smooth/1.5, # Downsampling it again
                peak_distance=20
            )

            aligned, peaks_aligned = peakSuppressor.align_by_topmost_peak(smoothed, peaks, topmost)

            # Panel of a few single A-scans with peak markers
            # fig, axes = plt.subplots(1, len(cols_to_show), figsize=(12, 3), tight_layout=True)

            # Big overview: all A-scans aligned into one image, with peaks overlaid
            ax[i,-2].imshow(enh,cmap='gray',aspect='auto',origin='upper')
            ax[i,-2].plot(ilm_smooth/1.5)
            suppressed_img = peakSuppressor.suppress_below_third_peak(enh,enh,peaks,margin=20,factor = 0.1)

            ax[i,-1].imshow(suppressed_img,cmap='gray',aspect='auto',origin='upper')


            for k, c in enumerate(cols_to_show):
                peakSuppressor.plot_ascan(ax[i,k], smoothed[:, c], peaks[c], title=f"col {c}")
                peakSuppressor.plot_ascan(ax[i,k], enh[:, c], peaks[c], title=f"raw col {c}",c='r')
                peakSuppressor.plot_ascan(ax[i,k], suppressed_img[:, c], peaks[c], title=f"suppressed raw col {c}",c='orange')
            
        fig.show()
        plt.show()



def guided_dp_rpe(
    img_or_prob,              # 2D array (H x W): pre-shadow enhancement (bright RPE) or prob in [0,1]
    guide_y=None,             # 1D array (W,) with np.nan where missing; columns with seeds/segments
    ilm_y=None,               # optional 1D array (W,) ILM row per col (to penalize above-ILM)
    use_prob=False,           # True if img_or_prob is already a probability map for RPE
    eps=1e-6,

    # DP move model
    max_step=3,               # allowed vertical step per column
    lambda_step=0.5,          # penalty *per column* for |Δrow| (smoothness)

    # Priors / costs
    alpha_guide=6.0,          # strength of attraction to guide
    sigma_guide=2.0,          # vertical spread (px) of guide attraction
    beta_depth=0.0,           # favor deeper rows: subtract beta_depth * (i/H) from cost
    ilm_margin=15,             # forbid (or strongly penalize) above ILM + margin
    ilm_max_factor=50.0,         # penalty for being above ILM+margin (use large number)

    # Hard window around guide (optional)
    hard_window=None,         # e.g., 4 to allow only ±4 px around guide in those columns; None disables
    hard_penalty=np.inf,       # what to add outside the hard window (np.inf = hard constraint)

    ONH_region=None, # a matrix same size as img_or_prob (could be just a vector that we expand, if need be). Will set cost to some constant lower value
    # As long as lambda_step != 0, should connect to correct layer hopefully? 
    ONH_value_factor = 0.5,


):
    """
    Returns:
      y_path: (W,) row indices of the DP-optimal path
      C:      (H,W) final cumulative cost map (useful for debugging)
    """
    I = np.asarray(img_or_prob, float)
    H, W = I.shape

    # --- Base data cost (lower near RPE)
    if use_prob:
        # cost = -log(p)  (peaks at RPE → low cost)
        data_cost = -np.log(np.clip(I, eps, 1.0))
    else:
        # normalize intensity to [0,1], then cost = 1 - norm (bright = low cost)
        mn, mx = np.nanmin(I), np.nanmax(I)
        norm = (I - mn) / (mx - mn + eps)
        data_cost = 1.0 - norm

    # --- Guidance reward: Gaussian tube around guide_y (negative cost)
    guide_cost = np.zeros_like(data_cost)
    if guide_y is not None:
        guide_y = np.asarray(guide_y, float)
        cols = np.flatnonzero(np.isfinite(guide_y))
        if cols.size:
            rows = np.arange(H)[:, None]         # (H,1)
            gy   = guide_y[cols][None, :]        # (1,K)
            # Gaussian reward centered at guide rows
            tube = np.exp(-0.5 * ((rows - gy) / max(sigma_guide, 1e-6))**2)
            # place into full image
            guide_cost[:, cols] -= alpha_guide * tube
            # Optional HARD window: disallow far from guide where it exists
            if hard_window is not None:
                hard_mask = np.ones((H, cols.size), dtype=bool)
                lo = np.clip((guide_y[cols] - hard_window).astype(int), 0, H-1)
                hi = np.clip((guide_y[cols] + hard_window).astype(int), 0, H-1)
                for k, (a, b) in enumerate(zip(lo, hi)):
                    hard_mask[a:b+1, k] = False  # False = inside allowed window
                # add penalty outside window
                data_cost[:, cols] += hard_penalty * hard_mask

    def apply_ilm_margin(cost, ilm_y, margin_below=4,max_factor=2,use_max=False):
        """
        Modify a cost map so that all rows above (ILM - margin_below) are set
        to a fixed high value (max of the current cost matrix).
        
        Parameters
        ----------
        cost : (H, W) ndarray
            Existing cost map.
        ilm_y : (W,) array_like
            ILM row positions (NaN where undefined).
        margin_below : int
            Margin (pixels) to extend below the ILM. Everything shallower
            than (ILM + margin_below) is forbidden by setting to max cost.
        """
        H, W = cost.shape
        ilm_y = np.asarray(ilm_y, float)
        max_cost = np.nanmax(cost)

        for j in np.flatnonzero(np.isfinite(ilm_y)):
            cutoff = int(min(H-1, max(0, np.floor(ilm_y[j] + margin_below))))
            if use_max:
                cost[:cutoff, j] = max_cost  # clamp everything shallower
            elif max_factor:
                cost[:cutoff, j] += max_cost//max_factor # clamp everything shallower
            else:
                raise Exception
        
        return cost


    # --- Depth prior: favor deeper rows (RPE is bottom-most among the three)
    # cost += -beta_depth * (i/H); i increases downward, so deeper rows get lower cost
    # if beta_depth != 0.0:
    #     depth = -beta_depth * (np.arange(H) / max(H-1, 1))[:, None]
    # else:
    #     depth = 0.0

    # cost = data_cost + guide_cost + ilm_cost + depth
    cost = data_cost + guide_cost 
    
    if ONH_region is not None:
        # ONH_region=None, # a matrix same size as img_or_prob (could be just a vector that we expand, if need be). Will set cost to some constant lower value
        # As long as lambda_step != 0, should connect to correct layer hopefully? 
        # this will need to be a hyperparameter
        if type(ONH_region)!=np.array:
            ONH_region = ONH_region.compute() # assuming it's dask at this point, maybe zarr
        # ONH_without_fovea = ONH_region.copy()
        # ONH_without_fovea[ONH_without_fovea!=1] = 0 
        ONH_without_fovea = np.where(ONH_region == 2, 0, ONH_region)

        to_assign = np.median(cost)*ONH_value_factor
        cost[ONH_without_fovea[:cost.shape[0],:].astype(bool)]=to_assign


    # if ilm_y is not None:
    cost = apply_ilm_margin(cost,ilm_y=ilm_y,margin_below=ilm_margin,max_factor=ilm_max_factor)
    # cost =  ilm_cost 
    y_path,C = run_DP_on_cost_matrix(cost,max_step,lambda_step)

    # Return the actual best path, the cost matrix C as determined by traversal of the DP matrix, and the raw cost image
    return y_path.astype(float), C, cost

def modify_cost_with_ONH_info(cost,ONH_region,ONH_value_factor):
    """overlying the optic nerve, make a constant cost so as to minimize disruption with DP 
    cost is the image to be input into the DP. ONH_region must be of the same shape."""
    if type(ONH_region)!=np.array:
        ONH_region = ONH_region.compute() # assuming it's dask at this point, maybe zarr
    # ONH_without_fovea = ONH_region.copy()
    # ONH_without_fovea[ONH_without_fovea!=1] = 0 
    ONH_without_fovea = np.where(ONH_region == 2, 0, ONH_region)

    to_assign = np.median(cost)*ONH_value_factor
    cost[ONH_without_fovea[:cost.shape[0],:].astype(bool)]=to_assign
    return cost


def run_DP_on_cost_matrix(cost,max_step,lambda_step):
    # --- Dynamic programming (first-order smoothness with max_step)
    H,W = cost.shape # assume 2D

    C = np.full((H, W), np.inf, dtype=float)
    P = np.full((H, W), -1, dtype=int)  # predecessor row index

    C[:, 0] = cost[:, 0]
    for j in range(1, W):
        # for each target row i at column j, consider predecessors k in [i-max_step .. i+max_step]
        for i in range(H):
            k0 = max(0, i - max_step)
            k1 = min(H - 1, i + max_step)
            prev = C[k0:k1+1, j-1] + lambda_step * np.abs(np.arange(k0, k1+1) - i)
            krel = int(np.argmin(prev))
            C[i, j] = cost[i, j] + prev[krel]
            P[i, j] = k0 + krel

    # --- Backtrack best path
    y_path = np.zeros(W, dtype=int)
    y_path[-1] = int(np.argmin(C[:, -1]))
    for j in range(W-1, 0, -1):
        y_path[j-1] = P[y_path[j], j]
    return y_path,C

def tube_smoother_DP(
    img,              # 2D array (H x W): pre-shadow enhancement (bright RPE) or prob in [0,1]
    guide_y=None,             # 1D array (W,) with np.nan where missing; columns with seeds/segments
    sigma_guide=2.0,          # vertical spread (px) of guide attraction
    eps=1e-6,

    # DP move model
    max_step=3,               # allowed vertical step per column
    lambda_step=0.5,          # penalty *per column* for |Δrow| (smoothness)
    ):
    """What this accomplishes: the enf_f image results in a segmentation above the RPE at the EZ, then by going back to blurred original image, 
    we get the RPE segmented as the region close enough to the EZ."""
    tubed_img = img.copy()
    H,W = img.shape

    if guide_y is not None:
        guide_y = np.asarray(guide_y, float)
        cols = np.flatnonzero(np.isfinite(guide_y))
        if cols.size:
            rows = np.arange(H)[:, None]         # (H,1)
            gy   = guide_y[cols][None, :]        # (1,K)
            # Gaussian reward centered at guide rows
            tube = np.exp(-0.5 * ((rows - gy) / max(sigma_guide, 1e-6))**2)
            mn_tube, mx_tube = np.nanmin(tube), np.nanmax(tube)
            norm_tube = (tube - mn_tube) / (mx_tube - mn_tube + eps)
            tubed_img[:, cols] *=  norm_tube

    mn, mx = np.nanmin(tubed_img), np.nanmax(tubed_img)
    norm = (tubed_img - mn) / (mx - mn + eps)
    cost = 1.0 - norm
    y_path,C = run_DP_on_cost_matrix(cost,max_step,lambda_step)

    return y_path.astype(float), C, cost


def upsample_path(path_ds, vertical_factor = None, horizontal_factor=None, original_length=512):
    """
    Upsample a down-sampled path by linear interpolation.

    Parameters
    ----------
    path_ds : 1D array, shape (W_ds,)
        Row indices at each down-sampled column.
    factor : float, optional
        The down-sampling factor (W_orig/W_ds).  Either factor or original_length must be provided.
    original_length : int, optional
        Desired length of the up-sampled path (W_orig). Overrides factor if both given.

    Returns
    -------
    path_orig : 1D float array, shape (W_orig,)
        Linear interpolation of path_ds onto the original column grid.
    """
    path_ds = path_ds.astype(np.float64)
    if vertical_factor:
        path_ds *= vertical_factor
    W_ds = len(path_ds)
    if original_length is None:
        if factor is None:
            raise ValueError("Need either factor or original_length")
        W_orig = int(round(W_ds * factor))
    else:
        W_orig = int(original_length)
        factor = W_orig / W_ds

    # x positions of the down-sampled samples on the original grid
    x_ds   = np.arange(W_ds) * factor
    # target x positions
    x_orig = np.arange(W_orig)

    # linear interp (extrapolation if needed, but here x_ds spans [0, W_orig))
    path_orig = np.interp(x_orig, x_ds, path_ds)

    return path_orig

def recalculate_single_seeded_cols(seeds_image,peak_suppressed_image,enh_image):
    """identifies the columns"""
    cols_to_recalculate = np.sum(seeds_image,axis=0)==1
    peak_suppressed_recalculated = peak_suppressed_image.copy()
    peak_suppressed_recalculated[:,cols_to_recalculate] = enh_image[:,cols_to_recalculate]
    return peak_suppressed_recalculated



import numpy as np

# 

# def rpe_hypersmoother(img):
#     """1_3_26: Heavy blur/downsample -> build cost (prefer high values) -> DP path -> flatten to that path.
#     Then we return the full y_f y_path so we can later un-hyper-smooth this. """
#     import code_files.segmentation_code.segmentation_plot_utils as spu
#     AB = spu.ArrayBoard()
#     AB.add(img,title = 'input img')

#     rigidities = [20,40,60]
#     for sig in [3,5,9]:
#         ds_y = 8; ds_x =  8
#         # sig  =  6.0
#         max_step    =  5
#         lambda_step =  0
#         fill = 0.0

#         # get image
#         H, W = img.shape

#         # blur helper (scipy -> cv2 -> box fallback)
#         def _gblur(a, sigma):
#             k = int(np.ceil(sigma * 6)) | 1
#             return cv2.GaussianBlur(a.astype(np.float32, copy=False), (k, k), sigma, borderType=cv2.BORDER_REPLICATE)
#         # coarse view
#         coarse = img[::ds_y, ::ds_x]
#         # AB.add(coarse,title = "downsampled")
#         coarse = _gblur(coarse, sig)
#         Hc, Wc = coarse.shape

#         # DP on "high values" => minimize negative intensity
#         cost = -coarse.astype(np.float32, copy=False)
#         y_dp, C = run_DP_on_cost_matrix(cost, max_step=max_step, lambda_step=lambda_step)
#         y_dp_smooth_list = [smooth_rpe_line(y_dp,rigidity=rigidity) for rigidity in rigidities] # Use this, it looks better!

#         # y_dp_smooth_less = smooth_rpe_line(y_dp,polyorder=7) # Use this, it looks better!
#         # AB.add(C,{'y_dp':y_dp},title = "cost_DP")
#         # AB.add(cost,{'y_dp':y_dp},"raw_cost")

#         # upsample path to full width + scale y
#         x_c = np.arange(Wc, dtype=np.float32)
#         x_f = np.linspace(0, Wc-1, W, dtype=np.float32)
#         y_f = np.interp(x_f, x_c, y_dp.astype(np.float32)).astype(np.float32) * ds_y
#         y_f_smooth_list = [np.interp(x_f, x_c, y_dp_smooth.astype(np.float32)).astype(np.float32) * ds_y 
#                            for y_dp_smooth in y_dp_smooth_list]
#         y0  = float(np.median(y_f))
#         shift = (y0 - y_f).astype(np.float32)  # + => shift down
#         shift_smooth_list = [(y0 - y_f_smooth).astype(np.float32) for 
#                              y_f_smooth in y_f_smooth_list]  # + => shift down

#         # shift-warp columns
#         y = np.arange(H, dtype=np.float32)
#         flat2 = np.empty((H, W), dtype=np.float32)
#         flat2_smooth_list = [np.empty((H, W), dtype=np.float32) for i in range(len(shift_smooth_list))]
#         imgf = img.astype(np.float32, copy=False)
#         for j in range(W):
#             src = y - float(shift[j])
#             flat2[:, j] = np.interp(src, y, imgf[:, j], left=fill, right=fill)

#             for i in range(len(shift_smooth_list)):
#                 src_smooth = y - float(shift_smooth_list[i][j])
#                 flat2_smooth_list[i][:, j] = np.interp(src_smooth, y, imgf[:, j], left=fill, right=fill)

#         AB.add(coarse,lines = {'y_dp':y_dp} | {f'rigiditiy = {rigidity}':y_f_smooth_list[i] for i,rigidity in enumerate(rigidities)},title = f"gaussian_blurred sig = {sig}")
#         # AB.add(coarse,lines = {'y_dp':y_dp},title = f"gaussian_blurred sig = {sig}")
#         # AB.add(img,lines={'y_f':y_f},title='original_with_overlay')
#         # AB.add(img,lines={'y_f':smooth_rpe_line(y_f)},title='original_with_smoothed')
#         AB.add(flat2,lines={},title=f'flattened sig = {sig}')
#         for k,flat2_smooth in enumerate(flat2_smooth_list):
#             AB.add(flat2_smooth,lines={},title=f'smoother flattened sig = {sig},rigidity = {rigidities[k]}')
#         # AB.add(flat2_smooth_less,lines={},title=f'smooth_less flattened sig = {sig}')

#     AB.render()
#     return flat2,y_f

# Commented out 1/9/26
# def rpe_hypersmoother(img):
#     """1_3_26: Heavy blur/downsample -> build cost (prefer high values) -> DP path -> flatten to that path.
#     Then we return the full y_f y_path so we can later un-hyper-smooth this. """

#     rigidity = 40
#     sig = 4.0
#     ds_y = 8; ds_x =  8
#     # sig  =  6.0
#     max_step    =  5
#     lambda_step =  0
#     fill = 0.0

#     # get image
#     H, W = img.shape

#     # blur helper (scipy -> cv2 -> box fallback)
#     def _gblur(a, sigma):
#         k = int(np.ceil(sigma * 6)) | 1
#         return cv2.GaussianBlur(a.astype(np.float32, copy=False), (k, k), sigma, borderType=cv2.BORDER_REPLICATE)
#     # coarse view
#     coarse = img[::ds_y, ::ds_x]
#     # AB.add(coarse,title = "downsampled")
#     coarse = _gblur(coarse, sig)
#     Hc, Wc = coarse.shape

#     # DP on "high values" => minimize negative intensity
#     cost = -coarse.astype(np.float32, copy=False)
#     y_dp, C = run_DP_on_cost_matrix(cost, max_step=max_step, lambda_step=lambda_step)
#     y_dp = smooth_rpe_line(y_dp,rigidity=40)

#     # upsample path to full width + scale y
#     x_c = np.arange(Wc, dtype=np.float32)
#     x_f = np.linspace(0, Wc-1, W, dtype=np.float32)
#     y_f = np.interp(x_f, x_c, y_dp.astype(np.float32)).astype(np.float32) * ds_y
#     y0  = float(np.median(y_f))
#     shift = (y0 - y_f).astype(np.float32)  # + => shift down

#     # shift-warp columns
#     y = np.arange(H, dtype=np.float32)
#     flat2 = np.empty((H, W), dtype=np.float32)
#     imgf = img.astype(np.float32, copy=False)
#     for j in range(W):
#         src = y - float(shift[j])
#         flat2[:, j] = np.interp(src, y, imgf[:, j], left=fill, right=fill)

#     return flat2,y_f


# def rpe_hypersmoother(img, ctx=None):
#     """Heavy blur/downsample -> cost -> DP path -> (optional) smooth path -> flatten to that path.
#     Returns: flat_img, y_path_full (float32, len=W)
#     Stores: ctx.hypersmoother_params (if ctx provided)
#     """
#     rigidity = 40
#     sig = 4.0
#     ds_y = 8; ds_x = 8
#     max_step = 5
#     lambda_step = 0
#     fill = 0.0

#     H, W = img.shape

#     def _gblur(a, sigma):
#         k = int(np.ceil(sigma * 6)) | 1
#         return cv2.GaussianBlur(
#             a.astype(np.float32, copy=False),
#             (k, k),
#             sigma,
#             borderType=cv2.BORDER_REPLICATE
#         )

#     # coarse view
#     coarse = img[::ds_y, ::ds_x]
#     coarse = _gblur(coarse, sig)
#     Hc, Wc = coarse.shape

#     # DP on "high values" => minimize negative intensity
#     cost = -coarse.astype(np.float32, copy=False)
#     y_dp, C = run_DP_on_cost_matrix(cost, max_step=max_step, lambda_step=lambda_step)
#     y_dp = smooth_rpe_line(y_dp, rigidity=rigidity)

#     # upsample path to full width + scale y
#     x_c = np.arange(Wc, dtype=np.float32)
#     x_f = np.linspace(0, Wc - 1, W, dtype=np.float32)
#     y_path_full = np.interp(x_f, x_c, y_dp.astype(np.float32)).astype(np.float32) * ds_y

#     target_y = float(np.median(y_path_full))
#     shift_y_full = (target_y - y_path_full).astype(np.float32)  # + => shift DOWN during flatten

#     # flatten (column warp)
#     y = np.arange(H, dtype=np.float32)
#     flat = np.empty((H, W), dtype=np.float32)
#     imgf = img.astype(np.float32, copy=False)
#     for j in range(W):
#         src = y - float(shift_y_full[j])
#         flat[:, j] = np.interp(src, y, imgf[:, j], left=fill, right=fill)

#     hypersmoother_params = {
#         "H": int(H),
#         "W": int(W),
#         "fill": float(fill),
#         "ds_y": int(ds_y),
#         "ds_x": int(ds_x),
#         "blur_sigma": float(sig),
#         "max_step": int(max_step),
#         "lambda_step": float(lambda_step),
#         "rigidity": float(rigidity),
#         "cost_shape": (int(Hc), int(Wc)),
#         "y_dp_coarse": y_dp.astype(np.float32, copy=False),
#         "C": C,  # leave as-is (could be big)
#         "y_path_full": y_path_full.astype(np.float32, copy=False),
#         "target_y": float(target_y),
#         "shift_y_full": shift_y_full.astype(np.float32, copy=False),
#     }

#     return flat, hypersmoother_params


def flatten_to_path(img, y_path_full, *, fill=0.0, target_y=None):
    """
    Column-warp img so y_path_full becomes horizontal at target_y (median by default).

    img: (H,W)
    y_path_full: (W,) float, y location per column in *img coordinates*
    Returns: flat_img (float32), shift_y_full (float32), target_y (float)
    """
    H, W = img.shape
    y_path_full = np.asarray(y_path_full, dtype=np.float32)
    if target_y is None:
        target_y = float(np.median(y_path_full))

    shift_y_full = (target_y - y_path_full).astype(np.float32)  # + => shift DOWN

    y = np.arange(H, dtype=np.float32)
    imgf = img.astype(np.float32, copy=False)
    flat = np.empty((H, W), dtype=np.float32)
    for j in range(W):
        src = y - float(shift_y_full[j])
        flat[:, j] = np.interp(src, y, imgf[:, j], left=fill, right=fill)

    return flat, shift_y_full, float(target_y)


class HypersmoothPreprocessors(object):
    @staticmethod
    def _gblur(a, sigma):
        k = int(np.ceil(sigma * 6)) | 1
        return cv2.GaussianBlur(a.astype(np.float32, copy=False), (k, k), sigma,
                                borderType=cv2.BORDER_REPLICATE)

    @staticmethod
    def _gradient(a):
        return _normalized_axial_gradient(a,20,dark2bright=True)
        


def rpe_hypersmoother_DP(img, 
                                        rigidity = 40,
                                        ds_y = 8,
                                        ds_x = 8,
                                        max_step = 5,
                                        lambda_step = 0,
                                        preprocess_function = HypersmoothPreprocessors._gblur,
                                        preprocess_kwargs = {'sigma' : 4.0},
                                        ctx=None):
    """Heavy blur/downsample -> cost -> DP path -> smooth path -> (later) flatten to that path.
    Returns: coarse image and path run upon that coarse image, after upsampling. Later function will flatten it hypersmoother_params
    """
    # fill = 0.0

    H, W = img.shape

    coarse = preprocess_function(img[::ds_y, ::ds_x], **preprocess_kwargs).astype(np.float32, copy=False)
    coarse /= (coarse.max() + 1e-7)
    Hc, Wc = coarse.shape

    cost = -coarse
    y_dp, C = run_DP_on_cost_matrix(cost, max_step=max_step, lambda_step=lambda_step)
    y_dp = smooth_rpe_line(y_dp, rigidity=rigidity)

    x_c = np.arange(Wc, dtype=np.float32)
    x_f = np.linspace(0, Wc - 1, W, dtype=np.float32)
    y_path_full = (np.interp(x_f, x_c, y_dp.astype(np.float32)).astype(np.float32) * ds_y)

    return coarse,y_path_full,y_dp



# def rpe_hypersmoother(img, ctx=None):
#     """Heavy blur/downsample -> cost -> DP path -> smooth path -> flatten to that path.
#     Returns: flat_img, hypersmoother_params
#     """
#     rigidity = 40
#     sig = 4.0
#     ds_y = 8; ds_x = 8
#     max_step = 5
#     lambda_step = 0
#     fill = 0.0

#     H, W = img.shape

#     def _gblur(a, sigma):
#         k = int(np.ceil(sigma * 6)) | 1
#         return cv2.GaussianBlur(a.astype(np.float32, copy=False), (k, k), sigma,
#                                 borderType=cv2.BORDER_REPLICATE)

#     coarse = _gblur(img[::ds_y, ::ds_x], sig)
#     Hc, Wc = coarse.shape

#     cost = -coarse.astype(np.float32, copy=False)
#     y_dp, C = run_DP_on_cost_matrix(cost, max_step=max_step, lambda_step=lambda_step)
#     y_dp = smooth_rpe_line(y_dp, rigidity=rigidity)

#     x_c = np.arange(Wc, dtype=np.float32)
#     x_f = np.linspace(0, Wc - 1, W, dtype=np.float32)
#     y_path_full = (np.interp(x_f, x_c, y_dp.astype(np.float32)).astype(np.float32) * ds_y)

#     flat, shift_y_full, target_y = flatten_to_path(img, y_path_full, fill=fill)

#     hypersmoother_params = {
#         "H": int(H), "W": int(W),
#         "fill": float(fill),
#         "ds_y": int(ds_y), "ds_x": int(ds_x),
#         "blur_sigma": float(sig),
#         "max_step": int(max_step),
#         "lambda_step": float(lambda_step),
#         "rigidity": float(rigidity),
#         "cost_shape": (int(Hc), int(Wc)),
#         "y_dp_coarse": y_dp.astype(np.float32, copy=False),
#         "C": C,
#         "y_path_full": y_path_full.astype(np.float32, copy=False),
#         "target_y": float(target_y),
#         "shift_y_full": shift_y_full.astype(np.float32, copy=False),
#     }

#     if ctx is not None:
#         ctx.hypersmoother_params = hypersmoother_params

#     return flat, hypersmoother_params


def rpe_unhypersmooth(flat_img, hs_params):
    """Invert the flattening warp (back to original geometry).
    flat_img must be same (H,W) that shift_y_full corresponds to.
    """
    H, W = flat_img.shape
    shift = hs_params["shift_y_full"].astype(np.float32, copy=False)
    fill = float(hs_params.get("fill", 0.0))

    assert len(shift) == W, "shift_y_full must match flat_img width"
    # (optional sanity) assert hs_params["H"] == H and hs_params["W"] == W

    y = np.arange(H, dtype=np.float32)
    out = np.empty((H, W), dtype=np.float32)
    flatf = flat_img.astype(np.float32, copy=False)

    # inverse: original(y,j) = flat(y + shift[j], j)
    for j in range(W):
        src = y + float(shift[j])
        out[:, j] = np.interp(src, y, flatf[:, j], left=fill, right=fill)

    return out


def warp_line_by_shift(y_line, warper_shift_y_full, direction="to_flat"):
    """
    Adjust ANY other 1D line (len=W) by the hypersmoother shift.

    y_line: array-like, shape (W,), y coords in ORIGINAL frame by default.
    direction:
        - "to_flat": map original->flattened coordinates  (subtract shift)
        - "to_orig": map flattened->original coordinates  (add shift)

    Example (ILM line):
        ilm_flat = warp_line_by_hypersmooth(ilm_orig, hs, "to_flat")
        ilm_back = warp_line_by_hypersmooth(ilm_flat, hs, "to_orig")
    """
    y_line = np.asarray(y_line, dtype=np.float32)
    shift = warper_shift_y_full.astype(np.float32, copy=False)

    assert y_line.shape[0] == shift.shape[0], "y_line must match width W"

    if direction == "to_flat":
        return y_line + shift
    elif direction == "to_orig":
        return y_line - shift
    else:
        raise ValueError("direction must be 'to_flat' or 'to_orig'")

def diff_boundary_enhance_and_blur_horiz(img,
                                        # down_params = {'vertical_kernel_size':25,'blur_ksize':20},
                                        down_vertical_kernel_size = 25,
                                        down_blur_ksize = 20,
                                        # up_params = {'vertical_kernel_size':15,'blur_ksize':40},
                                        up_vertical_kernel_size = 15,
                                        up_blur_ksize = 40,
                                        down_hblur = 40,
                                        up_hblur = 50,
                                        stack_for_AB = False
                                        ):
        enh_down = _boundary_enhance(img,down_vertical_kernel_size,dark2bright=False,blur_ksize=down_blur_ksize)
        enh_up = _boundary_enhance(img,up_vertical_kernel_size,dark2bright=True,blur_ksize=up_blur_ksize)
        hblur_down = _blur_image(enh_down,blur_k_x=down_hblur,blur_k_y=1,blur_type='blur')
        hblur_up = _blur_image(enh_up,blur_k_x=up_hblur,blur_k_y=1,blur_type='blur')

        hblur_down /= (hblur_down.max() + 1e-6)
        hblur_up /= (hblur_up.max() + 1e-6)
        diff_down_up = hblur_down - hblur_up
        diff_down_up[diff_down_up<0] = 0
        diff_down_up /= (diff_down_up.max() + 1e-6)

        diff_up_down =  hblur_up - hblur_down 
        diff_up_down[diff_up_down<0] = 0
        diff_up_down /= (diff_up_down.max() + 1e-6)

        # return np.vstack([np.hstack([hblur_down,hblur_up]),np.hstack([diff_down_up,diff_up_down])])
        # return diff_down_up
        if stack_for_AB:
            return np.vstack([np.hstack([hblur_down,hblur_up]),np.hstack([diff_down_up,diff_up_down])])
        else:
            return diff_down_up,diff_up_down,hblur_down,hblur_up


def high_res_tube_rpe_finder(flat_img,rpe_smooth_guide,highres_tube_sigma):
    """the guide here is th eoutput of my former algorithem, that gives an allowed zone, where things are then set to zero outside and a gradient is taken"""
    
    
    AB = spu.ArrayBoard(skip=True)
    # _boundary_enhance()
    # _boundary_enhance(flat_img,ver)
    # spu.sweep_to_arrayboard(AB,_boundary_enhance,base_kwargs={'img':flat_img}, grid = {'vertical_kernel_size':[15,25],'dark2bright':[True,False],'blur_ksize':[1,20,40]})


    AB.add(flat_img,title='flat image raw')

    
                                   # Thinking ideal will be down at 40, up at 50 or so. 


    diff_down_up,diff_up_down,hblur_down,hblur_up = diff_boundary_enhance_and_blur_horiz(flat_img,down_hblur=40,up_hblur=50)
    print("creating array board")
    AB2 = spu.ArrayBoard(skip=True,plt_display=False,dpi = 600) # will save it
    AB2.add(flat_img,lines={'rpe_guide_in':rpe_smooth_guide},title='input')
    AB2.add(flat_img,title='with line')
    spu.sweep_to_arrayboard(AB2,diff_boundary_enhance_and_blur_horiz,base_kwargs={'img':flat_img,'stack_for_AB':True}, 
                            grid = {'down_vertical_kernel_size':[5,10],
                                   'up_vertical_kernel_size':[10,20],
                                   })
    AB2.render()
    # for title,img in zip( ["diff_down_up","diff_up_down","hblur_down","hblur_up"], [diff_down_up,diff_up_down,hblur_down,hblur_up]):
    #     AB2.add(img,title=title)
    # AB2.render()

    AB.add(diff_down_up,title='diff_down_up')
    gaussian_tubed = apply_gaussian_tube_mul(diff_down_up,rpe_smooth_guide,highres_tube_sigma,gain=1,hard_window=40)
    AB.add(gaussian_tubed,title='gaussian_tubed')
    lower_edge_of_tubed = _normalized_axial_gradient(gaussian_tubed,vertical_kernel_size=4,dark2bright=True)
    AB.add(lower_edge_of_tubed,title='lower_edge_of_tubed')
    # tube_RPE_line,_ = run_DP_on_cost_matrix(1-gaussian_tubed,max_step=3,lambda_step=0)
    lower_edge_line,_ = run_DP_on_cost_matrix(1-lower_edge_of_tubed,max_step=3,lambda_step=0)
    # AB.add(gaussian_tubed,lines = {'new_rpe_line':lower_edge_line,'tube_rpe_line':tube_RPE_line,'old_rpe_line':rpe_smooth_guide},title='gaussian_tubed')
    AB.add(gaussian_tubed,lines = {'new_rpe_line':lower_edge_line,'old_rpe_line':rpe_smooth_guide},title='gaussian_tubed')
    AB.add(flat_img,lines = {'lower_edge_line':lower_edge_line,'old_rpe_line':rpe_smooth_guide},title='flat image raw')


    # Nice pattern here! Just postprocess add the lines you want
    # for triple in AB.items:
    #     # triple[1] = {'rpe_guide':rpe_smooth_guide}
    #     triple[1] = {'original_rpe_guide':rpe_smooth_guide,'new_rpe_line':new_RPE_line}


    



    # vks_false_factor= 0.75
    # vks_true_factor= 1.33
    # k_rows_factor= 1.0
    # k_cols_factor= 1.0
    # blur_ksize: int = 25

    # vks_false = 2 * round((32 / d_vertical) * vks_false_factor)
    # vks_true  = 2 * round((128 / d_vertical) * vks_true_factor)
    # k_rows    = 2 * round((20 / d_vertical) * k_rows_factor)
    # k_cols    = 2 * round(30 * k_cols_factor)


    # def enh_out_fn(flat_img,**kwargs):
    #     # out = compute_enh_diff( flat_img, blur_ksize=blur_ksize, vks_false=vks_false, vks_true=vks_true, k_rows=k_rows, k_cols=k_cols,)['diff']
    #     out = compute_enh_diff( img=flat_img,**kwargs)['diff']
    #     return out['diff']
    # spu.sweep_to_arrayboard(AB,enh_out_fn,base_kwargs={'flat_img':flat_img},grid={'blur_ksize':[25,40],'hard_window':[40,50]})

    """
    spu.sweep_to_arrayboard(AB,apply_gaussian_tube_mul,base_kwargs={'img':flat_img,'guide_y':rpe_smooth_guide},grid={"sigma":[20,30,40],'gain':[1,5,10],'hard_window':[40,50]})
    """
    # spu.sweep_to_arrayboard(AB,apply_gaussian_tube_mul,base_kwargs={'img':flat_img,'guide_y':rpe_smooth_guide},grid={"sigma":[3,5,10,20],'gain':[1,5,10]})
    # spu.sweep_to_arrayboard(AB,apply_gaussian_tube_mul,base_kwargs={'img':flat_img,'guide_y':rpe_smooth_guide,'fill':0},grid={"hard_window":[3,5,10,20]})
    # tubed_img = apply_gaussian_tube_mul(flat_img,guide_y=rpe_smooth_guide,sigma=highres_tube_sigma)
    # AB.add(tubed_img)
    AB.render()
    return diff_down_up,lower_edge_of_tubed,lower_edge_line


def sweeper_fn(images_dict=None,rpe_raw=None,rpe_smooth_guide=None,
               down_hblur=40, up_hblur=50, down_vertical_kernel_size=25, up_vertical_kernel_size=15, # Really big sweep params
               peak_sigma=1,peak_distance=5,peak_prominance=0.05, # peak params, don't matter much except sigma it seems
               hysteresis_high=0.1,hysteresis_low=0.01, # These are quite sensitive,  like a 0.2,0.01 is probably ballpark good
               factor=0.5, # Our suppression multiplier for suppressing upward
               lambda_step=0.2, 


               tube2_sigma = 20,
               tube2_hard_window=30,
               lambda_step2 = 0.4,
               min_flat_threshold=3,
               guide_reward = 1.2,


               highfreq_down_hblur=40, highfreq_up_hblur=50, highfreq_down_vertical_kernel_size=25, highfreq_up_vertical_kernel_size=15, # Really big sweep params


               really_big_sweep=False):

    if really_big_sweep:
        print("using the big sweep for the entire highres pipeline")
        flat_img = images_dict['flat_img']
        diff_down_up,diff_up_down,hblur_down,hblur_up = diff_boundary_enhance_and_blur_horiz(flat_img,
                                                                            down_hblur=down_hblur,
                                                                            up_hblur=up_hblur,
                                                                            down_vertical_kernel_size=down_vertical_kernel_size,
                                                                            up_vertical_kernel_size=up_vertical_kernel_size)
        gaussian_tubed = apply_gaussian_tube_mul(diff_down_up,rpe_smooth_guide,sigma=30,gain=1,hard_window=40)
        lower_edge_of_tubed = _normalized_axial_gradient(gaussian_tubed,vertical_kernel_size=4,dark2bright=True)


    else:
        print("using the smaller sweep starting at the peak finding")
        (lower_edge_of_tubed,
        flat_img,
        diff_down_up,
        diff_up_down,hblur_down,hblur_up,
        gaussian_tubed) = [images_dict[k] for k in ['lower_edge_of_tubed', 'flat_img', 'diff_down_up', 'diff_up_down','hblur_down','hblur_up', 'gaussian_tubed']]



    peaks = peakSuppressor.EZ_RPE_peak_suppression_pipeline(lower_edge_of_tubed,lower_edge_of_tubed,ilm_line=None,**{'sigma':peak_sigma,
                                                                                                                        'peak_distance':peak_distance,
                                                                                                                        'peak_prominance':peak_prominance}) # Start at top of img
    seeds = peaks_to_seeds(peaks,lower_edge_of_tubed.shape[0]) # convert o the boolean matrix

    OUT_peak_img = spu.overlay_peaks_on_image(lower_edge_of_tubed,peaks=peaks)

    probs,edge = ssf.helperFunctions._rpe_paths_prob_edge(path_trace_image=lower_edge_of_tubed,
                                                        seeds = seeds,
                                                        trace_neighborhood=1,
                                                        hysteresis_high=hysteresis_high,
                                                        hysteresis_low=hysteresis_low)

    rpe_raw = _extract_topbottom_line(edge,skip_extreme=0,direction='bottom')
    suppressed = peakSuppressor.suppress_above_below_line(lower_edge_of_tubed,rpe_raw,factor=factor)
    rpe_refined,_ = run_DP_on_cost_matrix(1-suppressed,max_step=2,lambda_step=lambda_step)

    # Sadly a failed experiment
    truncated_rpe_refined = keep_only_flat_segments(rpe_refined,slope_tol=0.5,min_len=min_flat_threshold) # Really this isn't quite what we want. 
    # tubed_flat_img = apply_gaussian_tube_mul(flat_img,rpe_refined,sigma=tube2_sigma,gain=1,hard_window=tube2_hard_window)
    # guide_img = apply_guideline_discount(tubed_flat_img,guideline=truncated_rpe_refined,factor=guide_reward,band=1)
    # rpe_refined2,_ = run_DP_on_cost_matrix(1-guide_img,max_step=2,lambda_step=lambda_step2) # Already a pretty dang good output

    # Next try is to use the rpe_refined, following truncation, as a starting point for similarity seed tracing, on various gradient str
    highfreq_diff_down_up,highfreq_diff_up_down,highfreq_hblur_down,highfreq_hblur_up = diff_boundary_enhance_and_blur_horiz(flat_img,
                                                                        down_hblur=highfreq_down_hblur,
                                                                        up_hblur=highfreq_up_hblur,
                                                                        down_vertical_kernel_size=highfreq_down_vertical_kernel_size,
                                                                        up_vertical_kernel_size=highfreq_up_vertical_kernel_size)


    tubed_highres_grad = apply_gaussian_tube_mul(highfreq_diff_down_up,rpe_refined,sigma=tube2_sigma,gain=1,hard_window=tube2_hard_window) # Tighten the sigma
    lower_edge_of_highres_grad_tubed = _normalized_axial_gradient(tubed_highres_grad,vertical_kernel_size=4,dark2bright=True)
    guide_img = apply_guideline_discount(lower_edge_of_highres_grad_tubed,guideline=truncated_rpe_refined,factor=guide_reward,band=1)
    rpe_refined2,_ = run_DP_on_cost_matrix(1-guide_img,max_step=2,lambda_step=lambda_step2) # Already a pretty dang good output


    # Another failed experiment -- seed tracing with similarity: Would need a ton of blurring to get better, going to fail, as this img is so not clean.
    # seed_line = rpe_refined
    # tubed_highres_grad = apply_gaussian_tube_mul(highfreq_diff_down_up,rpe_refined,sigma=tube2_sigma,gain=1,hard_window=tube2_hard_window)
    # rpe_refined_seeds = line_to_seeds(seed_line,H=highfreq_diff_down_up.shape[0])
    # paths = _trace_paths( highfreq_diff_down_up, rpe_refined_seeds, neighbourhood=1)
    # prob = _probability_image(paths, highfreq_diff_down_up.shape)
    # prob /= (prob.max() + 1e-6)
    # y_final = prob.argmax(axis=0).astype(np.int32)   # length W





    outputs = []
    for output in [suppressed,flat_img]:
        output = output/np.amax(output)
        output = spu.overlay_line_on_image(output,rpe_smooth_guide,color=[0,0,1])
        output = spu.overlay_line_on_image(output,rpe_raw,color=[0,1,0])
        output = spu.overlay_line_on_image(output,rpe_refined,color=[1,0,0])
        # output = spu.overlay_line_on_image(output,truncated_rpe_refined,color=[1,1,0])
        outputs.append(output)

    # output_matrix = [
    #     [hblur_down,hblur_up,diff_up_down,diff_down_up],
    #     [OUT_peak_img,edge,suppressed, outputs[0]],
    #     [guide_img,spu.overlay_line_on_image(guide_img,rpe_refined2), outputs[1],spu.overlay_line_on_image(flat_img,rpe_refined2)]
    # ]

    # output_matrix = [
    #     [hblur_down,hblur_up,diff_up_down,diff_down_up],
    #     [OUT_peak_img,edge,suppressed, outputs[0]],
    #     [highfreq_diff_down_up,spu.overlay_line_on_image(spu.overlay_line_on_image(tubed_highres_grad,seed_line),y_final,color=[0,1,1]),prob,spu.overlay_line_on_image(outputs[1],y_final,color=[0,1,1])]
    # ]

    output_matrix = [
        [hblur_down,hblur_up,diff_up_down,diff_down_up],
        [OUT_peak_img,edge,suppressed, outputs[0]],
        [tubed_highres_grad,guide_img,spu.overlay_line_on_image(spu.overlay_line_on_image(guide_img,truncated_rpe_refined),rpe_refined2,color=[0,1,1]),spu.overlay_line_on_image(outputs[1],rpe_refined2,color=[0,1,1])]
    ]



    output_matrix = [[spu.to_rgb(img) for img in img_list] for img_list in output_matrix]
    # output_shapes = [[img.shape for img in img_list] for img_list in output_matrix]
    # print(output_shapes)
    
    
    return np.vstack([np.hstack(l) for l in output_matrix])


def diff_enh_sweeper_fn(original_image,rpe_hypersmooth_params,rpe_smooth_guide,which_smoother,down_hblur=40, up_hblur=50, down_vertical_kernel_size=25, up_vertical_kernel_size=15,**kwargs):

    rpe_unwarped = warp_line_by_shift(rpe_smooth_guide,rpe_hypersmooth_params['shift_y_full'],direction='to_orig')
    if which_smoother == "rpe_smooth":
        # print(rpe_smooth_guide)
        smoother = rpe_unwarped
        # print(smoother)
    elif which_smoother == "hypersmooth_line":
        smoother = rpe_hypersmooth_params['y_path_full']
    else: raise Exception

    flat_img, shift_y_full, target_y = flatten_to_path(original_image,smoother)



    diff_down_up,diff_up_down,hblur_down,hblur_up = diff_boundary_enhance_and_blur_horiz(flat_img,
                                                                                         down_hblur=down_hblur,
                                                                        up_hblur=up_hblur,
                                                                        down_vertical_kernel_size=down_vertical_kernel_size,
                                                                        up_vertical_kernel_size=up_vertical_kernel_size)

    output_matrix = [
            [spu.overlay_line_on_image(original_image,smoother,thickness=3),flat_img],
            [diff_down_up,diff_up_down],
            [hblur_down,hblur_up]
    ] 
    output_matrix = [[spu.to_rgb(img) for img in img_list] for img_list in output_matrix]
    return np.vstack([np.hstack(l) for l in output_matrix])

def debug_high_res_tube_rpe_finder(flat_img,original_image,rpe_hypersmooth_params,rpe_smooth_guide,highres_tube_sigma,**diff_boundary_enhance_kwargs):
    """A version that will finish the pipeline basically to allow sweeps. 
    This itself is the function to go into the AB sweep, with stacked outputs"""
    
    if diff_boundary_enhance_kwargs is None:
       diff_boundary_enhance_kwargs  = {
        'down_hblur':40,
        'up_hblur':50,
        'down_vertical_kernel_size':25,
        'up_vertical_kernel_size':15,
       }

    
    # diff_down_up,diff_up_down,hblur_down,hblur_up = diff_boundary_enhance_and_blur_horiz(flat_img,down_hblur=40,up_hblur=50)
    print("creating array board")
    # AB2 = spu.ArrayBoard(skip=False,plt_display=False,dpi = 600,save_tag='_testing_flatteners') # will save it
    # AB2.add(original_image,lines={'rpe_guide_in':rpe_smooth_guide},title='input')
    # # AB2.add(flat_img,title='with line')
    # spu.sweep_to_arrayboard(AB2,diff_enh_sweeper_fn,base_kwargs={'original_image':original_image,
    #                                                               'rpe_hypersmooth_params': rpe_hypersmooth_params,
    #                                                              'rpe_smooth_guide':rpe_smooth_guide,
    #                                                               'highres_tube_sigma':highres_tube_sigma,
    # }, 
    #                         grid = {'which_smoother':['rpe_smooth','hypersmooth_line'],
    #                             'down_hblur':[10,40],
    #                             'up_hblur':[10,50],
    #                             'down_vertical_kernel_size':[5,10,25],
    #                                },nworkers=8)
    # AB2.render()

  

    # raise Exception

    diff_down_up,diff_up_down,hblur_down,hblur_up = diff_boundary_enhance_and_blur_horiz(flat_img,**diff_boundary_enhance_kwargs)

    gaussian_tubed = apply_gaussian_tube_mul(diff_down_up,rpe_smooth_guide,highres_tube_sigma,gain=1,hard_window=40)
    lower_edge_of_tubed = _normalized_axial_gradient(gaussian_tubed,vertical_kernel_size=4,dark2bright=True)
    # peak_suppressed = peakSuppressor.EZ_RPE_peak_suppression_pipeline(lower_edge_of_tubed,lower_edge_of_tubed,ilm_line=None) # Start at top of img
    AB1 = spu.ArrayBoard(dpi=500,plt_display=False,save_tag="_peak_suppression_pipeline",skip=True)
    AB1.add(lower_edge_of_tubed,title = "lower_edge_of_tubed")

    _ = spu.sweep_to_arrayboard(AB1,lambda **kw: spu.overlay_peaks_on_image(lower_edge_of_tubed,peakSuppressor.EZ_RPE_peak_suppression_pipeline(**kw)),base_kwargs={ 'peak_source_image':lower_edge_of_tubed, 'image_to_alter':lower_edge_of_tubed, 'ilm_line':None,},
                            grid = { 'sigma':[0.1,1],'peak_distance':[2,5], 'peak_prominance':[0.01,0.05,0.2]})

    AB1.render()

    peaks = peakSuppressor.EZ_RPE_peak_suppression_pipeline(lower_edge_of_tubed,lower_edge_of_tubed,ilm_line=None,**{'sigma':1,'peak_distance':5, 'peak_prominance':0.05}) # Start at top of img
    seeds = peaks_to_seeds(peaks,lower_edge_of_tubed.shape[0]) # convert o the boolean matrix



    AB = spu.ArrayBoard(dpi=500,plt_display=False,save_tag="_EZ_RPE_tracings",skip=True)
    probs,edge = ssf.helperFunctions._rpe_paths_prob_edge(path_trace_image=lower_edge_of_tubed,
                                                          seeds = seeds,
                                                          trace_neighborhood=1,
                                                          hysteresis_high=0.1,
                                                          hysteresis_low=0.01)
    # AB.add(np.vstack([probs,edge]),title='probs/edge')



    AB.add(lower_edge_of_tubed,title = "lower_edge_of_tubed")
    AB.add(seeds,title = "seeds")
    _ = spu.sweep_to_arrayboard(AB,lambda **kw: np.vstack(ssf.helperFunctions._rpe_paths_prob_edge(**kw)),base_kwargs={'path_trace_image':lower_edge_of_tubed, 'seeds' : seeds, 'trace_neighborhood':1,'nworkers':1},
                            grid = { 'hysteresis_high':[0.1,0.4], 'hysteresis_low':[0.01,0.02,0.09]})
    AB.render()


    AB_rpe=spu.ArrayBoard(dpi=500,plt_display=False,skip=True,save_tag='_rpe_raw_added')
    AB_rpe.add(edge,title='rpe_raw_overlay')
    _ = spu.sweep_to_arrayboard(AB_rpe,lambda **kw: spu.overlay_line_on_image(lower_edge_of_tubed,_extract_topbottom_line(**kw)),
                                base_kwargs={'edge_img':edge},
                                grid = {'skip_extreme':[0,1,2]})
    AB_rpe.render()

    rpe_raw = _extract_topbottom_line(edge,skip_extreme=0,direction='bottom')

    from datetime import datetime

    AB_suppressed_up=spu.ArrayBoard(dpi=500,plt_display=False,skip=False,save_tag=f'_suppressed_up_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}') #don't overwrite for a bit
    AB_suppressed_up.add(spu.overlay_line_on_image(lower_edge_of_tubed,rpe_raw),title='input')


    images_dict = { 'diff_down_up':diff_down_up, 'gaussian_tubed':gaussian_tubed, 'lower_edge_of_tubed':lower_edge_of_tubed, 'flat_img':flat_img}
    _ = spu.sweep_to_arrayboard(AB_suppressed_up,sweeper_fn,base_kwargs={'images_dict':images_dict, 'rpe_raw':rpe_raw,'rpe_smooth_guide':rpe_smooth_guide,'really_big_sweep':True},
                                grid = {'factor':[0,0.4],
                                        'peak_prominance':[0.05],
                                        "hysteresis_high":[0.1,0.3],
                                        # "hysteresis_high":[0.3],
                                        "hysteresis_low":[0.01,0.05],
                                        # "hysteresis_low":[0.05],
                                        "lambda_step":[0,0.1],
                                        },
                                        nworkers=8
                                        )
    AB_suppressed_up.render()


    print("complete")
    raise Exception

                                                          
    # probs = _trace_paths(lower_edge_of_tubed,seeds[:,::2]) # Choose half density seeds

    # Q_upper_edge_removed = _normalized_axial_gradient(gaussian_tubed,vertical_kernel_size=20,dark2bright=True)
    lower_edge_line,_ = run_DP_on_cost_matrix(1-lower_edge_of_tubed,max_step=3,lambda_step=0)
    line_overlaid = spu.overlay_line_on_image(lower_edge_of_tubed,lower_edge_line)
    line_overlaid = spu.overlay_line_on_image(line_overlaid,rpe_smooth_guide,color=np.array([0,1,0]))
    output_dict = {
                'diff_down_up':diff_down_up,
                'diff_up_down':diff_up_down,
                'hblur_down':hblur_down,
                'hblur_up':hblur_up,
                'gaussian_tubed':gaussian_tubed,
                'lower_edge_of_tubed':lower_edge_of_tubed,
                'lower_edge_line':lower_edge_line,
    }
    (hblur_down,hblur_up,diff_down_up,peak_suppressed,lower_edge_of_tubed,line_overlaid) = [spu.to_rgb(e) for e in [hblur_down,hblur_up,diff_down_up,peak_suppressed,lower_edge_of_tubed,line_overlaid]]
    output_grid = np.vstack([np.hstack([hblur_down,hblur_up]),
               np.hstack([diff_down_up,peak_suppressed]),
               np.hstack([lower_edge_of_tubed,line_overlaid])])
    # output_grid = np.hstack()
    return output_grid




def apply_gaussian_tube_mul(
    img: np.ndarray,
    guide_y: np.ndarray,
    sigma: float = 10,
    gain: float = 1.0,
    hard_window: int | None = None,
    fill: float = 0,

) -> np.ndarray:
    # REFACTOR: should use this in the other tube code
    """
    Multiply image by a Gaussian 'tube' centered on guide_y (one y per column).
      - tube(y,x) = 1 + gain * exp(-0.5 * ((y-guide_y[x])/sigma)^2)
      - if hard_window is set: outside |y-guide|>hard_window -> multiplier = fill

    img: (H,W) float or uint image
    guide_y: (W,) y-coordinates (row index) with np.nan for missing
    """
    H, W = img.shape
    gy = np.asarray(guide_y, float)
    cols = np.flatnonzero(np.isfinite(gy))
    if cols.size == 0:
        return img.copy()

    sigma = max(float(sigma), 1e-6)

    rows = np.arange(H)[:, None]          # (H,1)
    g = gy[cols][None, :]                 # (1,K)
    tube = np.exp(-0.5 * ((rows - g) / sigma) ** 2)   # (H,K)
    # above performs a broadcasting s.t. each index in a column is the value of the distance of that pixel from the guide_y, and then that's put thru the square

    mult = np.ones((H, W), dtype=float)
    mult[:, cols] = 0 + float(gain) * tube

    old_mult = mult.copy()
    if hard_window is not None:
        hw = int(hard_window)
        outside = np.abs(rows - g) > hw           # (H,K)
        r, k = np.where(outside)
        mult[r, cols[k]] = fill

    out = img.astype(float, copy=False) * mult

    mn, mx = np.nanmin(out), np.nanmax(out)
    out = (out - mn) / (mx - mn + 1e-6)
    # cost = 1.0 - norm
    # y_path,C = run_DP_on_cost_matrix(cost,max_step,lambda_step)

    return out

def keep_only_flat_segments(y, slope_tol=0.5, min_len=20):
    """
    Returns y_out same shape as y, with NaN everywhere except contiguous "flat" segments
    where |diff(y)| <= slope_tol and segment length >= min_len.
    """
    y = np.asarray(y, dtype=float)
    dy = np.diff(y)
    flat = np.abs(dy) <= slope_tol

    edges = np.flatnonzero(np.diff(np.r_[False, flat, False]))
    runs = edges.reshape(-1, 2)  # runs over dy indices [a,b)

    out = np.full_like(y, np.nan, dtype=float)
    for a, b in runs:
        x0 = a
        x1 = b  # inclusive in y
        if (x1 - x0 + 1) >= min_len:
            out[x0:x1+1] = y[x0:x1+1]

    if np.all(np.isnan(out)):
        raise ValueError("No flat segments found (try larger slope_tol or smaller min_len).")
    return out


def apply_guideline_discount(img, guideline, factor=0.5, band=0):
    """
    img: (H,W) where HIGHER is better (e.g., postiive response)
    guideline: (W,) y per column; NaN means "no guide" for that column
    factor: multiply cost by this (>1 makes guided pixels even cheaper once you put into 1-img cost fmt)
    band: also apply to +/- band rows around guideline
    """
    img = img.copy()
    H, W = img.shape
    g = np.asarray(guideline, dtype=float)
    cols = np.flatnonzero(~np.isnan(g))
    rows = np.round(g[cols]).astype(int)
    rows = np.clip(rows, 0, H - 1)

    for t in range(-band, band + 1):
        rr = np.clip(rows + t, 0, H - 1)
        img[rr, cols] *= factor

    return img



# 2/2/26, starting to experimenet with true GS methods

import numpy as np

def gs_single_surface_pymaxflow(
    score_img: np.ndarray,
    smoothness: float = 0.5,
    row_range: tuple[int, int] | None = None,
    normalize: bool = True,
    eps: float = 1e-8,
    inf: float = 1e12,
    return_debug: bool = False,
):
    """
    Single-surface graph cut (s-t mincut) using PyMaxflow.

    Parameters
    ----------
    score_img : (H, W) array
        Higher means "surface likely here" (e.g., gradient magnitude / edge score).
    smoothness : float
        Potts coupling strength between neighboring columns (larger => smoother surface).
    row_range : (r0, r1) or None
        If provided, restrict to rows [r0, r1) to speed up / focus search; output y is in full-image coords.
    normalize : bool
        If True, normalize score_img to [0, 1] for stability.
    eps : float
        Numerical stability.
    inf : float
        Large capacity for "infinite" constraints (column prefix closure).
    return_debug : bool
        If True, also returns (flow, segments_bool, score_used).

    Returns
    -------
    y : (W,) int array
        Surface row index per column (0..H-1). If a column has no selected nodes (rare),
        y[c] becomes r0-1 (i.e., one row above the search band).
    """
    try:
        import maxflow
    except ImportError as e:
        raise ImportError(
            "PyMaxflow not installed. Try: pip install PyMaxflow"
        ) from e

    if score_img.ndim != 2:
        raise ValueError(f"score_img must be 2D (H,W). Got shape {score_img.shape}")

    H_full, W = score_img.shape

    # Restrict rows if desired
    if row_range is None:
        r0, r1 = 0, H_full
    else:
        r0, r1 = row_range
        r0 = int(max(0, r0))
        r1 = int(min(H_full, r1))
        if r1 <= r0:
            raise ValueError(f"Invalid row_range {row_range} after clipping => [{r0}, {r1})")

    s = score_img[r0:r1, :].astype(np.float64, copy=False)
    H = s.shape[0]

    # Optional normalization to [0,1]
    if normalize:
        s_min = float(np.nanmin(s))
        s_max = float(np.nanmax(s))
        s = (s - s_min) / (s_max - s_min + eps)
        s = np.nan_to_num(s, nan=0.0, posinf=1.0, neginf=0.0)

    # Convert "boundary score" s(r,c) to node weights w(r,c) so that:
    # sum_{k=0..r} w(k,c) = s(r,c)
    w = np.empty_like(s)
    w[0, :] = s[0, :]
    w[1:, :] = s[1:, :] - s[:-1, :]

    # Build graph: maximum-weight closed set via s-t mincut
    g = maxflow.Graph[float]()
    nodes = g.add_grid_nodes((H, W))

    # Unary terms (node weights)
    # If w>0: edge Source->node with capacity w
    # If w<0: edge node->Sink with capacity -w
    cap_source = np.maximum(w, 0.0)
    cap_sink   = np.maximum(-w, 0.0)
    g.add_grid_tedges(nodes, cap_source, cap_sink)

    # Column prefix closure:
    # If (r,c) is selected (in Source set), then (r-1,c) must be selected too.
    # Enforce with directed "infinite" edges: (r,c) -> (r-1,c)
    for r in range(1, H):
        # vectorize over columns
        for c in range(W):
            g.add_edge(int(nodes[r, c]), int(nodes[r - 1, c]), inf, 0.0)

    # Horizontal Potts smoothness:
    # Encourage same label between (r,c) and (r,c+1) across all rows
    # This couples the prefixes and smooths y[c].
    if smoothness > 0:
        lam = float(smoothness)
        for c in range(W - 1):
            for r in range(H):
                u = int(nodes[r, c])
                v = int(nodes[r, c + 1])
                g.add_edge(u, v, lam, lam)

    flow = g.maxflow()

    # get_grid_segments returns a boolean mask of nodes in the SOURCE set (per PyMaxflow convention).
    seg = g.get_grid_segments(nodes)  # shape (H,W), bool

    # Extract surface: last selected row in each column.
    # If no rows selected in a column, we return r0-1 for that column.
    y_local = np.full((W,), -1, dtype=np.int32)
    for c in range(W):
        col = seg[:, c]
        if np.any(col):
            y_local[c] = int(np.max(np.flatnonzero(col)))
        else:
            y_local[c] = -1

    y = y_local + r0  # map back to full-image coordinates

    if return_debug:
        return y, flow, seg, s
    return y


# import numpy as np

# def gs_single_surface_pymaxflow_simple(
#     score_img: np.ndarray,
#     smoothness: float = 0.5,
#     row_range: tuple[int, int] | None = None,
#     normalize: bool = True,
#     tau: float | None = None,          # baseline; if None we auto-pick
#     segments_are_sink: bool = True,    # common in some pymaxflow builds
#     eps: float = 1e-8,
#     inf: float = 1e12,
#     return_debug: bool = False,
# ):
#     """
#     Single-surface (one y per column) via s-t mincut.
#     Uses w = score - tau so the boundary lands near ridges.
#     """
#     import maxflow

#     if score_img.ndim != 2:
#         raise ValueError(f"score_img must be 2D (H,W). Got shape {score_img.shape}")

#     H_full, W = score_img.shape
#     if row_range is None:
#         r0, r1 = 0, H_full
#     else:
#         r0, r1 = map(int, row_range)
#         r0 = max(0, r0); r1 = min(H_full, r1)
#         if r1 <= r0:
#             raise ValueError(f"Invalid row_range => [{r0},{r1})")

#     s = score_img[r0:r1, :].astype(np.float64, copy=False)
#     H = s.shape[0]

#     if normalize:
#         smin, smax = float(np.nanmin(s)), float(np.nanmax(s))
#         s = (s - smin) / (smax - smin + eps)
#         s = np.nan_to_num(s, nan=0.0, posinf=1.0, neginf=0.0)

#     # Key step: make background slightly negative so "select all rows" is NOT optimal.
#     if tau is None:
#         # robust baseline: a bit above median
#         tau = float(np.quantile(s, 0.60))
#     w = s - tau  # (H,W), can be positive near ridge, negative elsewhere

#     g = maxflow.Graph[float]()
#     nodes = g.add_grid_nodes((H, W))

#     # unary terms from w
#     cap_source = np.maximum(w, 0.0)
#     cap_sink   = np.maximum(-w, 0.0)
#     g.add_grid_tedges(nodes, cap_source, cap_sink)

#     # prefix-from-top closure: if (r,c) is in SOURCE, then (r-1,c) must be in SOURCE.
#     for r in range(1, H):
#         for c in range(W):
#             g.add_edge(int(nodes[r, c]), int(nodes[r - 1, c]), inf, 0.0)

#     # horizontal Potts smoothness
#     if smoothness > 0:
#         lam = float(smoothness)
#         for c in range(W - 1):
#             for r in range(H):
#                 u = int(nodes[r, c])
#                 v = int(nodes[r, c + 1])
#                 g.add_edge(u, v, lam, lam)

#     flow = g.maxflow()
#     seg = g.get_grid_segments(nodes)  # bool mask (either SOURCE or SINK depending on build)

#     # Convert to SOURCE mask if necessary
#     src = (~seg) if segments_are_sink else seg

#     # Extract boundary as last SOURCE row in each column
#     y_local = np.full((W,), -1, dtype=np.int32)
#     for c in range(W):
#         col = src[:, c]
#         if np.any(col):
#             y_local[c] = int(np.max(np.flatnonzero(col)))
#         else:
#             y_local[c] = -1

#     y = y_local + r0

#     if return_debug:
#         return y, flow, src, s, w, tau
#     return y

def gs_single_surface_pymaxflow_simple(
    score_img: np.ndarray,
    smoothness: float = 0.5,
    row_range: tuple[int, int] | None = None,
    normalize: bool = True,
    tau_q: float = 0.75,              # per-column quantile baseline
    tau: float | None = None,
    force_top: bool = True,
    segments_are_sink: bool = True,   # flip if prefix-check fails
    eps: float = 1e-8,
    inf: float = 1e12,
    return_debug: bool = False,
):
    import maxflow

    if score_img.ndim != 2:
        raise ValueError(f"score_img must be 2D (H,W). Got shape {score_img.shape}")

    H_full, W = score_img.shape
    if row_range is None:
        r0, r1 = 0, H_full
    else:
        r0, r1 = map(int, row_range)
        r0 = max(0, r0); r1 = min(H_full, r1)
        if r1 <= r0:
            raise ValueError(f"Invalid row_range => [{r0},{r1})")

    s = score_img[r0:r1, :].astype(np.float64, copy=False)
    # from scipy.ndimage import gaussian_filter
    # s = s-gaussian_filter(s,sigma=(8,1))

    H = s.shape[0]

    if normalize:
        smin, smax = float(np.nanmin(s)), float(np.nanmax(s))
        s = (s - smin) / (smax - smin + eps)
        s = np.nan_to_num(s, nan=0.0, posinf=1.0, neginf=0.0)

    # per-column baseline so background doesn't dominate
    if tau is None: # use ta
        tau = np.quantile(s, tau_q, axis=0)          # shape (W,)
    else:
        tau = np.full((W,),tau)
    w = s - tau[None, :]                         # shape (H,W)

    g = maxflow.Graph[float]()
    nodes = g.add_grid_nodes((H, W))

    cap_source = np.maximum(w, 0.0)
    cap_sink   = np.maximum(-w, 0.0)
    g.add_grid_tedges(nodes, cap_source, cap_sink)

    if force_top:
        g.add_grid_tedges(nodes[0, :], inf, 0.0)  # force at least one selected per column

    # prefix closure: x[r,c]=1 => x[r-1,c]=1
    for r in range(1, H):
        for c in range(W):
            g.add_edge(int(nodes[r, c]), int(nodes[r - 1, c]), inf, 0.0)

    # horizontal smoothness
    if smoothness > 0:
        lam = float(smoothness)
        for c in range(W - 1):
            for r in range(H):
                u = int(nodes[r, c])
                v = int(nodes[r, c + 1])
                g.add_edge(u, v, lam, lam)

    flow = g.maxflow()
    seg = g.get_grid_segments(nodes)

    # Convert seg -> SOURCE mask
    src = (~seg) if segments_are_sink else seg

    # extract boundary = last SOURCE row per column
    y_local = np.empty((W,), dtype=np.int32)
    for c in range(W):
        col = src[:, c]
        # force_top guarantees at least one True; still guard:
        y_local[c] = int(np.max(np.flatnonzero(col))) if np.any(col) else -1

    y = y_local + r0

    if return_debug:
        return y, flow, src, s, w, tau
    return y


import numpy as np

def gs_two_surfaces_pymaxflow(
    score1: np.ndarray,                 # (H,W) evidence for upper surface
    score2: np.ndarray,                 # (H,W) evidence for lower surface
    dmin: int = 5,                      # min separation: y2 >= y1 + dmin
    dmax: int = 40,                     # max separation: y2 <= y1 + dmax
    smooth1: float = 0.5,               # horizontal smoothness for surface1
    smooth2: float = 0.5,               # horizontal smoothness for surface2
    row_range: tuple[int, int] | None = None,
    normalize: bool = True,
    tau_factor1: float = 0.75,               # per-column baseline quantile for score1
    tau_factor2: float = 0.75,               # per-column baseline quantile for score2
    tau_mode: str='quantile',
    depth_penalty1: float = 0.001,
    depth_penalty2: float = 0.001,
    force_top: bool = True,             # force row0 selected for both surfaces
    segments_are_sink: bool = True,     # pymaxflow quirk: get_grid_segments may return sink set
    eps: float = 1e-8,
    inf: float = 1e12,
    return_debug: bool = False,
):
    """
    Two-surface graph cut with ordering + separation bounds.

    Representation:
      Xk(r,c)=1 iff r <= yk[c] (top-prefix).
      yk[c] = last row with Xk=1 in column c.

    Constraints enforced via infinite-capacity implications:
      (A) Prefix:      Xk(r,c)=1 -> Xk(r-1,c)=1
      (B) Ordering:    X1(r,c)=1 -> X2(r,c)=1  (=> y1 <= y2)
      (C) Min sep:     X1(r,c)=1 -> X2(r+dmin,c)=1  (=> y2 >= y1+dmin)
      (D) Max sep:     X2(r,c)=1 -> X1(r-dmax,c)=1  (=> y2 <= y1+dmax)

    Returns:
      y1, y2 (W,)
      optionally debug: flow, src1, src2, s1, s2, w1, w2, tau1, tau2
    """
    import maxflow

    if score1.shape != score2.shape:
        raise ValueError(f"score1 and score2 must have same shape. Got {score1.shape} vs {score2.shape}")
    if score1.ndim != 2:
        raise ValueError("score1/score2 must be 2D (H,W).")

    H_full, W = score1.shape

    # Restrict rows if desired
    if row_range is None:
        r0, r1 = 0, H_full
    else:
        r0, r1 = map(int, row_range)
        r0 = max(0, r0)
        r1 = min(H_full, r1)
        if r1 <= r0:
            raise ValueError(f"Invalid row_range => [{r0},{r1})")

    s1 = score1[r0:r1, :].astype(np.float64, copy=False)
    s2 = score2[r0:r1, :].astype(np.float64, copy=False)
    H = s1.shape[0]

    if normalize:
        def _norm(a):
            amin, amax = float(np.nanmin(a)), float(np.nanmax(a))
            a = (a - amin) / (amax - amin + eps)
            return np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=0.0)
        s1 = _norm(s1)
        s2 = _norm(s2)

    # Per-column baselines -> membership weights w = score - tau
    if tau_mode == 'quantile':
        tau1 = np.quantile(s1, tau_factor1, axis=0)  # (W,)
        tau2 = np.quantile(s2, tau_factor2, axis=0)  # (W,)
    elif tau_mode == 'mean_percent':
        tau1 = np.mean(s1, axis=0)*tau_factor1  # (W,)
        tau2 = np.mean(s2, axis=0)*tau_factor2  # (W,)
    elif tau_mode == 'absolute':
        tau1 = np.full((W,),tau_factor1)
        tau2 = np.full((W,),tau_factor2)

    # w1 = s1 - tau1[None, :]
    # w2 = s2 - tau2[None, :]

    r = np.arange(H)[:, None] / (H - 1)
    w1 = s1 - tau1[None, :] - depth_penalty1 * r   # upper surface: prefer shallower
    w2 = s2 - tau2[None, :] + depth_penalty2  * r   # lower surface: prefer deeper

    g = maxflow.Graph[float]()
    nodes1 = g.add_grid_nodes((H, W))  # surface 1 nodes
    nodes2 = g.add_grid_nodes((H, W))  # surface 2 nodes

    # --- Unary terms (t-edges) ---
    g.add_grid_tedges(nodes1, np.maximum(w1, 0.0), np.maximum(-w1, 0.0))
    g.add_grid_tedges(nodes2, np.maximum(w2, 0.0), np.maximum(-w2, 0.0))

    if force_top:
        g.add_grid_tedges(nodes1[0, :], inf, 0.0)
        g.add_grid_tedges(nodes2[0, :], inf, 0.0)

    # --- Prefix closure inside each surface ---
    for r in range(1, H):
        for c in range(W):
            g.add_edge(int(nodes1[r, c]), int(nodes1[r - 1, c]), inf, 0.0)
            g.add_edge(int(nodes2[r, c]), int(nodes2[r - 1, c]), inf, 0.0)

    # --- Horizontal Potts smoothness inside each surface ---
    if smooth1 > 0:
        lam = float(smooth1)
        for c in range(W - 1):
            for r in range(H):
                u = int(nodes1[r, c])
                v = int(nodes1[r, c + 1])
                g.add_edge(u, v, lam, lam)

    if smooth2 > 0:
        lam = float(smooth2)
        for c in range(W - 1):
            for r in range(H):
                u = int(nodes2[r, c])
                v = int(nodes2[r, c + 1])
                g.add_edge(u, v, lam, lam)

    # --- Cross-surface constraints ---

    # (B) Ordering: X1(r,c)=1 -> X2(r,c)=1
    # Edge: node1(r,c) -> node2(r,c) with inf
    for r in range(H):
        for c in range(W):
            g.add_edge(int(nodes1[r, c]), int(nodes2[r, c]), inf, 0.0)

    # (C) Min separation: X1(r,c)=1 -> X2(r+dmin,c)=1
    if dmin > 0:
        for r in range(H - dmin):
            rr = r + dmin
            for c in range(W):
                g.add_edge(int(nodes1[r, c]), int(nodes2[rr, c]), inf, 0.0)

    # (D) Max separation: X2(r,c)=1 -> X1(r-dmax,c)=1
    if dmax is not None and dmax >= 0:
        for r in range(dmax, H):
            rr = r - dmax
            for c in range(W):
                g.add_edge(int(nodes2[r, c]), int(nodes1[rr, c]), inf, 0.0)

    flow = g.maxflow()

    seg1 = g.get_grid_segments(nodes1)
    seg2 = g.get_grid_segments(nodes2)

    src1 = (~seg1) if segments_are_sink else seg1
    src2 = (~seg2) if segments_are_sink else seg2

    # Extract y1,y2 as last SOURCE row per column
    y1_local = np.empty((W,), dtype=np.int32)
    y2_local = np.empty((W,), dtype=np.int32)
    for c in range(W):
        col1 = src1[:, c]
        col2 = src2[:, c]
        y1_local[c] = int(np.max(np.flatnonzero(col1))) if np.any(col1) else -1
        y2_local[c] = int(np.max(np.flatnonzero(col2))) if np.any(col2) else -1

    y1 = y1_local + r0
    y2 = y2_local + r0

    if return_debug:
        return y1, y2, flow, src1, src2, s1, s2, w1, w2, tau1, tau2
    return y1, y2

# After abandoning the above most likely, for reasons described in the notebook

import numpy as np

def run_two_surface_DP(
    cost1: np.ndarray,              # (H,W) cost for upper surface (lower is better)
    cost2: np.ndarray,              # (H,W) cost for lower surface
    dmin: int,
    dmax: int,
    max_step1: int,
    max_step2: int,
    lambda1: float,
    lambda2: float,
    return_tables: bool = False,
):
    """
    Parallel 2-surface DP with hard per-column separation bounds and first-order smoothness.

    Constraints (per column j):
      dmin <= y2[j] - y1[j] <= dmax
      |y1[j] - y1[j-1]| <= max_step1
      |y2[j] - y2[j-1]| <= max_step2

    Objective:
      sum_j cost1[y1[j], j] + cost2[y2[j], j]
        + lambda1 * sum_j |y1[j]-y1[j-1]|
        + lambda2 * sum_j |y2[j]-y2[j-1]|

    Returns:
      y1, y2 (W,) int arrays
      optionally: C, P1, P2 where
        C is (H,H,W) with inf for invalid pairs (prototype; memory heavy)
        P1,P2 are predecessor row indices for backtracking
    """
    from tqdm import tqdm
    if cost1.shape != cost2.shape:
        raise ValueError(f"cost1 and cost2 must match. Got {cost1.shape} vs {cost2.shape}")
    if cost1.ndim != 2:
        raise ValueError("cost1/cost2 must be 2D (H,W).")
    if dmin < 0 or dmax < dmin:
        raise ValueError("Require 0 <= dmin <= dmax.")

    H, W = cost1.shape
    # Debug: count non-finite costs
    n_bad1 = np.size(cost1) - np.isfinite(cost1).sum()
    n_bad2 = np.size(cost2) - np.isfinite(cost2).sum()
    if n_bad1 or n_bad2:
        print(f"[DP] non-finite costs: cost1={n_bad1}, cost2={n_bad2}")

    # Strongly recommended: treat NaN as +inf (unusable pixel)
    cost1 = np.nan_to_num(cost1, nan=np.inf, posinf=np.inf, neginf=np.inf)
    cost2 = np.nan_to_num(cost2, nan=np.inf, posinf=np.inf, neginf=np.inf)

    # We'll store DP over (i1,i2) pairs. For prototype clarity, we use dense (H,H) and mask invalids.
    C_prev = np.full((H, H), np.inf, dtype=float)
    C_cur  = np.full((H, H), np.inf, dtype=float)

    # predecessors (store prev rows for i1 and i2)
    P1 = np.full((H, H, W), -1, dtype=int)
    P2 = np.full((H, H, W), -1, dtype=int)

    # Initialize column 0
    for i1 in range(H):
        i2_lo = i1 + dmin
        i2_hi = min(H - 1, i1 + dmax)
        if i2_lo >= H:
            continue
        for i2 in range(i2_lo, i2_hi + 1):
            C_prev[i1, i2] = cost1[i1, 0] + cost2[i2, 0]
            # P1/P2 at j=0 stay -1
    if not np.isfinite(C_prev).any():
        print(np.unique(cost1[:,0]))
        print(np.unique(cost2[:,0]))
        print(H)
        raise RuntimeError(
            "No feasible initial states at column 0. "
            "Check dmin/dmax vs H and check cost1/cost2 for NaN/Inf at col 0."
        )

    # DP forward
    for j in tqdm(range(1, W)):
        C_cur.fill(np.inf)

        for i1 in range(H):
            i2_lo = i1 + dmin
            i2_hi = min(H - 1, i1 + dmax)
            if i2_lo >= H:
                continue

            # predecessor range for i1
            k1_lo = max(0, i1 - max_step1)
            k1_hi = min(H - 1, i1 + max_step1)

            for i2 in range(i2_lo, i2_hi + 1):
                # predecessor range for i2
                k2_lo = max(0, i2 - max_step2)
                k2_hi = min(H - 1, i2 + max_step2)

                best_val = np.inf
                best_k1 = -1
                best_k2 = -1

                # brute-force search in the local predecessor window
                # (validity of (k1,k2) enforced by checking C_prev[k1,k2] < inf)
                for k1 in range(k1_lo, k1_hi + 1):
                    step1 = lambda1 * abs(i1 - k1)

                    # for this k1, k2 must also satisfy separation at prev column:
                    # dmin <= k2 - k1 <= dmax  =>  k2 in [k1+dmin, k1+dmax]
                    kk2_lo = max(k2_lo, k1 + dmin)
                    kk2_hi = min(k2_hi, k1 + dmax)
                    if kk2_lo > kk2_hi:
                        continue

                    for k2 in range(kk2_lo, kk2_hi + 1):
                        prev_cost = C_prev[k1, k2]
                        if not np.isfinite(prev_cost):
                            continue
                        val = prev_cost + step1 + lambda2 * abs(i2 - k2)
                        if val < best_val:
                            best_val = val
                            best_k1 = k1
                            best_k2 = k2

                if best_k1 >= 0:
                    C_cur[i1, i2] = cost1[i1, j] + cost2[i2, j] + best_val
                    P1[i1, i2, j] = best_k1
                    P2[i1, i2, j] = best_k2

        # DEBUG EARLY DISCONNECT DETECTION
        if not np.isfinite(C_cur).any():
            # helpful extra diagnostics
            finite_prev = np.isfinite(C_prev).sum()
            finite_c1 = np.isfinite(cost1[:, j]).sum()
            finite_c2 = np.isfinite(cost2[:, j]).sum()
            raise RuntimeError(
                f"DP became infeasible at column {j}. "
                f"finite_prev_pairs={finite_prev}, finite_cost1_col={finite_c1}/{H}, finite_cost2_col={finite_c2}/{H}. "
                f"Try checking NaN/Inf in costs or that constraints/banding aren't implicitly applied upstream."
            )
        C_prev, C_cur = C_cur, C_prev  # swap buffers

    # DEBUG
    finite_mask = np.isfinite(C_prev)
    if not finite_mask.any():
        raise RuntimeError(
            "No feasible ending state (all costs are inf) at final column. "
            "This means the DP was infeasible by the last layer."
        )

    # pick min among finite entries only
    end_flat = np.argmin(np.where(finite_mask, C_prev, np.inf))
    end_i1, end_i2 = np.unravel_index(end_flat, C_prev.shape)


    # # Pick best ending state at column W-1
    # end_i1, end_i2 = np.unravel_index(np.argmin(C_prev), C_prev.shape)

    # Backtrack
    y1 = np.zeros(W, dtype=int)
    y2 = np.zeros(W, dtype=int)
    y1[-1] = int(end_i1)
    y2[-1] = int(end_i2)

    for j in range(W - 1, 0, -1):
        k1 = P1[y1[j], y2[j], j]
        k2 = P2[y1[j], y2[j], j]
        
    
        if k1 < 0 or k2 < 0:
            print("[DP] Backtrack failure diagnostics:")
            print("  j:", j)
            print("  state (y1,y2):", y1[j], y2[j])
            # Note: at backtrack time, C_prev no longer corresponds to column j.
            # But the pointer missing means this state was never assigned at column j.

            # If this happens, the DP got stuck (constraints too tight or costs all inf)
            raise RuntimeError(f"Backtrack failed at column {j}. Try relaxing constraints.")
        y1[j - 1] = k1
        y2[j - 1] = k2

    if return_tables:
        # If you really want full tables, we can reconstruct them, but it's memory heavy.
        # For now return the predecessor cubes and final C_prev snapshot.
        return y1, y2, C_prev.copy(), P1, P2

    return y1, y2
