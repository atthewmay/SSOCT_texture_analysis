import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from PyPDF2 import PdfMerger

from scipy.ndimage import convolve, maximum_filter1d
from scipy.signal import savgol_filter
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # adds Han_AIR/ to path
from visualization_utils import PlotTracer
p = PlotTracer(show=False)
p.tracing=True
from collections import Counter

from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
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


def _trace_paths(enh, seeds, neighbourhood=1,break_zeros=False):
    H, W = enh.shape
    offsets = np.arange(-neighbourhood, neighbourhood + 1)
    paths = []
    counter = Counter()
    for r0, c0 in zip(*np.nonzero(seeds)):

        # added: trace in both directions from each seed
        for dc in (+1, -1):
            path = [(r0, c0)]
            r, c = r0, c0
            contig_breaker=0
            while True:
                # guard next column
                if not (0 <= c + dc < W):
                    break

                # find candidate rows
                cand_r = r + offsets
                cand_r = cand_r[(cand_r >= 0) & (cand_r < H)]

                # choose next row by looking *ahead* at c+dc
                diffs = np.abs(enh[cand_r, c + dc] - enh[r, c])
                    

                # fast check: are all entries in diffs identical?
                # (ptp = max–min)
                if np.ptp(diffs) == 0:
                    # tie: pick the median candidate
                    if break_zeros:
                        if np.all(diffs==0):
                            contig_breaker += 1 
                            if contig_breaker > 10:
                                counter.update(['path_broken'])
                                break
                        else:
                            contig_breaker=0
                    r_new = int(cand_r[len(cand_r)//2])
                    counter.update(diffs)
                else:
                    # normal case: choose the minimum-difference row
                    idx = diffs.argmin()
                    r_new = int(cand_r[idx])
                    counter.update(["yes_difference"])

                # advance
                r, c = r_new, c + dc
                path.append((r, c))

            paths.append(path)
    return paths


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
    def peak_suppression_pipeline(peak_source_image,image_to_alter,ilm_line,**kwargs):
        """peak source img and img to alter are likely the same images"""
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

