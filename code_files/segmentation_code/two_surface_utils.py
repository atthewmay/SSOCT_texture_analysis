from textwrap import fill
import numpy as np
import code_files.segmentation_code.segmentation_utility_functions as suf
import code_files.segmentation_code.segmentation_plot_utils as spu
from code_files.segmentation_code.segmentation_utility_functions import barrier_sum
from code_files.segmentation_code.segmentation_utility_functions import barrier_from_cost_exp 


def _column_best_and_second_best_gap(cost_col: np.ndarray, guard: int = 3):
    """
    Returns:
      best_i: argmin index
      gap: (second_best_outside_guard - best)
    """
    H = cost_col.shape[0]
    best_i = int(np.argmin(cost_col))
    best_v = float(cost_col[best_i])

    lo = max(0, best_i - guard)
    hi = min(H, best_i + guard + 1)

    second = np.inf
    if lo > 0:
        v = float(np.min(cost_col[:lo]))
        if v < second:
            second = v
    if hi < H:
        v = float(np.min(cost_col[hi:]))
        if v < second:
            second = v

    gap = float(second - best_v)
    return best_i, gap


import numpy as np

def run_two_surface_DP(
    cost1: np.ndarray,              # (H,W) boundary cost for upper surface (lower is better)
    cost2: np.ndarray,              # (H,W) boundary cost for lower surface
    dmin: int,
    dmax: int,
    max_step1: int,
    max_step2: int,
    lambda1: float,
    lambda2: float,
    # --- NEW: encourage lower surface to "claim" the single ridge (via NMS peaks)
    prefer_lower_on_single: bool = True,
    single_kappa: float = 0.0,          # strength of anchoring penalty (0 disables)
    kappa_mode: str = 'abs_dist',          # strength of anchoring penalty (0 disables)
    # NMS params (expects "evidence" where higher=better)
    peak_sigma: float=1,
    peak_distance: float = 5,
    peak_prominance: float = 0.05, # again if sweep later, refactor into config
    # optional extras (usually leave 0 initially)
    sep_gamma: float = 0.0,             # softly prefer (y2-y1) near dmin
    depth_a1: float = 0.0,              # add depth_a1 * r to cost1 (r in [0,1])
    depth_a2: float = 0.0,              # add depth_a2 * r to cost2
    

    barrier_cost_params: dict = {'alpha':6.0}, # woud modify if using the other
    barrier_cost_function = barrier_from_cost_exp,
    darkness_barrier_factor: float = 0.0,

    # --- NEW: ONH handling ---
    ONH_region=None,
    onh_cost: float = 0.5,               # boundary cost in ONH columns
    # onh_disable_kappa: bool = True,      # ignore anchoring in ONH columns
    # onh_disable_barrier: bool = True,    # ignore darkness barrier in ONH columns
    # onh_exclude_from_norm: bool = True,  # exclude ONH columns from mn/mx for c2n

    # debug
    return_debug: bool = False,
):
    """
    Parallel 2-surface DP with hard separation + first-order smoothness.
    Adds an optional NMS-based "single ridge" detector that anchors y2 to the
    strongest peak in cost2-evidence when only one peak is present.

    Anchor penalty (only for columns classified as single-peak):
      extra = kappa[j] * |y2[j] - anchor2[j]|
    where kappa[j] is either single_kappa (single-peak) or 0 (multi-peak).

    Returns:
      y1, y2 (W,)
      if return_debug: (y1, y2, debug_dict)
    """
    # --------- basic checks ----------
    if cost1.shape != cost2.shape:
        raise ValueError(f"cost1 and cost2 must match. Got {cost1.shape} vs {cost2.shape}")
    if cost1.ndim != 2:
        raise ValueError("cost1/cost2 must be 2D (H,W).")
    if dmin < 0 or dmax < dmin:
        raise ValueError("Require 0 <= dmin <= dmax.")

    H, W = cost1.shape

    onh_cols = suf._onh_cols_from_region(ONH_region, W)

    cost1 = np.asarray(cost1, dtype=float)
    cost2 = np.asarray(cost2, dtype=float)
    cost1 = np.nan_to_num(cost1, nan=np.inf, posinf=np.inf, neginf=np.inf)
    cost2 = np.nan_to_num(cost2, nan=np.inf, posinf=np.inf, neginf=np.inf)

    # barrier = cost2 # shape (H,W), values in [0,1] ideally

    # --------- optional depth priors (small) ----------
    if depth_a1 != 0.0 or depth_a2 != 0.0:
        r = (np.arange(H, dtype=float) / max(H - 1, 1))[:, None]  # (H,1)
        if depth_a1 != 0.0:
            cost1 = cost1 + depth_a1 * r
        if depth_a2 != 0.0:
            cost2 = cost2 + depth_a2 * r

    # --------- NMS-based anchor2 + kappa[j] ----------
    anchor2 = np.zeros(W, dtype=np.int32)
    kappa = np.zeros(W, dtype=float)
    # nms_maxima2 = None

    # -------------make inverse of cost two -----------
    # c2c = cost2.copy()
    # finite = np.isfinite(c2c)
    # mn = float(np.min(c2c[finite])); mx = float(np.max(c2c[finite]))
    # c2n = (c2c - mn) / (mx - mn + 1e-8)     # 0=best, 1=worst
    # evidence2 = 1.0 - c2n        
    # Bcs = np.cumsum(evidence2, axis=0)          # (H,W)
    # # evidence2 = None


    # -------------new cost barrier fn --------
    c2 = cost2.copy()
    finite = np.isfinite(c2)
    mn = float(np.min(c2[finite])); mx = float(np.max(c2[finite]))
    c2n = (c2 - mn) / (mx - mn + 1e-8)   # 0=best ridge, 1=worst background
    evidence2 = 1.0 - c2n

    Bcs = None
    barrier = None
    if darkness_barrier_factor != 0.0:
        barrier = barrier_cost_function(c2n, **barrier_cost_params)         # option 1
        Bcs = np.cumsum(barrier, axis=0)

    if onh_cols is not None:
        if Bcs is not None:
            Bcs[:,onh_cols]=0
        cost1[:,onh_cols]=onh_cost
        cost2[:,onh_cols]=onh_cost
    # print("we are attempting to do the quickfig")
    # spu.quickfig(cost1)

    if prefer_lower_on_single and single_kappa > 0.0:
        # Build an evidence image from cost2 (higher=better for peaks).
        # Robust normalize cost2 -> [0,1], then evidence = 1 - norm_cost.
        # c2 = cost2.copy() # note that the input is likely alread per-col normed
        # finite = np.isfinite(c2)
        # if np.any(finite):
        #     mn = float(np.min(c2[finite]))
        #     mx = float(np.max(c2[finite]))
        #     c2n = (c2 - mn) / (mx - mn + 1e-8)
        # else:
        #     c2n = c2
        # evidence2 = 1.0 - c2n

        # --- you must have _nms_columnwise in scope (your function)
        # if
        # nms_maxima2 = suf._nms_columnwise(
        #     evidence2,
        #     radius=nms_radius,
        #     thresh=nms_thresh,
        #     vertical_filter=nms_vertical_filter,
        #     value_filter=nms_value_filter,
        #     keeptop=nms_keeptop,
        # )
        # print(nms_maxima2)


        peaks = suf.peakSuppressor.EZ_RPE_peak_suppression_pipeline(1-c2n,1-c2n,ilm_line=None,
                                                                    **{'sigma':peak_sigma,
                                                                        'peak_distance':peak_distance,
                                                                        'peak_prominance':peak_prominance}) # again if sweep later, refactor into config
        # import pdb; pdb.set_trace()
        # print(f" peaks is none? {peaks is None}")



        # classify each column as single-peak vs multi-peak (cluster flat-tops)
        for j in range(W):
            # rows = np.nonzero(peaks[:, j])[0]
            # rows = np.nonzero(peaks[j][:, j])[0]
            rows = peaks[j]
            if rows.size == 0:
                # fallback: take strongest evidence, but don't anchor hard
                anchor2[j] = int(np.argmax(evidence2[:, j]))
                kappa[j] = 0.0
                continue

            # pick strongest peak as anchor2
            vals = evidence2[rows, j]
            anchor2[j] = int(rows[int(np.argmax(vals))])

            # single peak => anchor on (kappa = single_kappa), else off
            kappa[j] = float(single_kappa) if len(rows) == 1 else 0.0 # 

    else:
        # no anchoring; still provide anchor2 for debug (argmin cost2)
        for j in range(W):
            anchor2[j] = int(np.argmin(cost2[:, j]))

    if onh_cols is not None:
        kappa[onh_cols] = 0.0
    # --------- DP tables ----------
    C_prev = np.full((H, H), np.inf, dtype=float)
    C_cur  = np.full((H, H), np.inf, dtype=float)
    P1 = np.full((H, H, W), -1, dtype=np.int32)
    P2 = np.full((H, H, W), -1, dtype=np.int32)

    # --------- init column 0 ----------
    j = 0
    for i1 in range(H):
        i2_lo = i1 + dmin
        i2_hi = min(H - 1, i1 + dmax)
        if i2_lo >= H:
            continue
        c1 = cost1[i1, j]
        if not np.isfinite(c1):
            continue

        for i2 in range(i2_lo, i2_hi + 1):
            c2 = cost2[i2, j]
            if not np.isfinite(c2):
                continue

            extra = 0.0
            if kappa[j] != 0.0:
                if kappa_mode == 'abs_dist':
                    extra += kappa[j] * abs(i2 - int(anchor2[j]))
                elif kappa_mode == 'reweight':
                    pass # will adjust below
                else:
                    raise Exception
            if sep_gamma != 0.0:
                extra += sep_gamma * abs((i2 - i1) - dmin)

            C_prev[i1, i2] = c1 + c2 + extra

    if not np.isfinite(C_prev).any():
        raise RuntimeError("No feasible initial states at column 0 (check dmin/dmax and inf costs).")

    # --------- forward DP ----------
    for j in range(1, W):
        C_cur.fill(np.inf)

        for i1 in range(H):
            i2_lo = i1 + dmin
            i2_hi = min(H - 1, i1 + dmax)
            if i2_lo >= H:
                continue

            c1 = cost1[i1, j]
            if not np.isfinite(c1):
                continue

            k1_lo = max(0, i1 - max_step1)
            k1_hi = min(H - 1, i1 + max_step1)

            for i2 in range(i2_lo, i2_hi + 1):
                c2 = cost2[i2, j]
                if kappa_mode == 'reweight':
                    c2 *= 1+kappa[j]

                if not np.isfinite(c2):
                    continue

                # per-column extras (anchor + optional sep preference)
                extra = 0.0
                if kappa[j] != 0.0:
                    if kappa_mode == 'abs_dist':
                        extra += kappa[j] * abs(i2 - int(anchor2[j]))
                    elif kappa_mode == 'reweight':
                        pass # will adjust below
                    else:
                        raise Exception

                if sep_gamma != 0.0:
                    extra += sep_gamma * abs((i2 - i1) - dmin)

                k2_lo = max(0, i2 - max_step2)
                k2_hi = min(H - 1, i2 + max_step2)

                best_val = np.inf
                best_k1 = -1
                best_k2 = -1

                for k1 in range(k1_lo, k1_hi + 1):
                    step1 = lambda1 * abs(i1 - k1)

                    kk2_lo = max(k2_lo, k1 + dmin)
                    kk2_hi = min(k2_hi, k1 + dmax)
                    if kk2_lo > kk2_hi:
                        continue

                    for k2 in range(kk2_lo, kk2_hi + 1):
                        prev_cost = C_prev[k1, k2]
                        if not np.isfinite(prev_cost):
                            continue
                        if darkness_barrier_factor != 0:
                            dy = abs(i2 - k2)
                            step2 = lambda2 * dy + darkness_barrier_factor * barrier_sum(Bcs,j, k2, i2)
                        else:
                            step2 = lambda2 * abs(i2 - k2)
                        val = prev_cost + step1 + step2




                        if val < best_val:
                            best_val = val
                            best_k1 = k1
                            best_k2 = k2

                if best_k1 >= 0:
                    C_cur[i1, i2] = c1 + c2 + extra + best_val
                    P1[i1, i2, j] = best_k1
                    P2[i1, i2, j] = best_k2

        if not np.isfinite(C_cur).any():
            raise RuntimeError(f"DP infeasible at column {j} (all inf). Check costs/constraints.")
        C_prev, C_cur = C_cur, C_prev

    # --------- choose best end state ----------
    finite_mask = np.isfinite(C_prev)
    if not finite_mask.any():
        raise RuntimeError("No feasible ending state (all inf).")
    end_flat = int(np.argmin(np.where(finite_mask, C_prev, np.inf)))
    end_i1, end_i2 = np.unravel_index(end_flat, C_prev.shape)

    # --------- backtrack ----------
    y1 = np.zeros(W, dtype=np.int32)
    y2 = np.zeros(W, dtype=np.int32)
    y1[-1] = int(end_i1)
    y2[-1] = int(end_i2)

    for j in range(W - 1, 0, -1):
        k1 = int(P1[y1[j], y2[j], j])
        k2 = int(P2[y1[j], y2[j], j])
        if k1 < 0 or k2 < 0:
            raise RuntimeError(f"Backtrack failed at column {j}.")
        y1[j - 1] = k1
        y2[j - 1] = k2

    if not return_debug:
        return y1.astype(int), y2.astype(int)

    debug = {
        "anchor2": anchor2.astype(int),
        "kappa": kappa.astype(float),
        "final_cost": float(C_prev[end_i1, end_i2]),
    }
    if peaks is not None:
        # print("peaks is not none, should be adding to the debug outptu")
        debug["peaks"] = peaks
    if barrier is not None:
        debug["barrier"] = barrier
        # debug["evidence2"] = evidence2
    return y1.astype(int), y2.astype(int), debug

def make_cost_from_img(img: np.ndarray, mode: str = "inv_colmax", eps: float = 1e-8) -> np.ndarray:
    """
    Turns an evidence image into a boundary cost matrix.
    - If img is 'evidence' (higher=better), cost should be lower where img is high.
    Modes:
      inv_colmax: normalize each column by its max, then cost = 1 - norm
      inv_global: global normalize then cost = 1 - norm
      neg: cost = -img (use if img already well-scaled)
    """
    x = np.asarray(img, dtype=float)
    if mode == "inv_colmax":
        den = np.max(x, axis=0, keepdims=True)
        norm = x / (den + eps)
        return 1.0 - norm
    elif mode == "inv_global":
        mn = np.min(x)
        mx = np.max(x)
        norm = (x - mn) / (mx - mn + eps)
        return 1.0 - norm
    elif mode == "neg":
        return -x
    else:
        raise ValueError(f"Unknown mode: {mode}")


def dp_two_surface_debug_metrics(cost1, cost2, y1, y2, debug=None):
    """
    Returns quick 1D diagnostic vectors you can plot:
      - separation d
      - anchor row for cost2 (best per col)
      - |y2 - anchor2|
      - per-col best cost2 and 2nd-best gap2 (if debug provided)
    """
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)
    d = y2 - y1

    # Always compute anchor2 as argmin(cost2) for “what peak is it aiming for”
    anchor2 = np.argmin(cost2, axis=0).astype(int)
    dist_to_anchor2 = np.abs(y2 - anchor2)

    out = {
        "d": d.astype(float),
        "anchor2": anchor2.astype(float),
        "dist_to_anchor2": dist_to_anchor2.astype(float),
    }
    if debug is not None:
        if "gap2" in debug: out["gap2"] = np.asarray(debug["gap2"], dtype=float)
        if "kappa" in debug: out["kappa"] = np.asarray(debug["kappa"], dtype=float)
        if "anchor2" in debug: out["anchor2_debug"] = np.asarray(debug["anchor2"], dtype=float)
    return out

from joblib import Parallel, delayed


import inspect

def sweep_two_surface_dp(param_list, n_jobs: int = 8):
    allowed = set(inspect.signature(run_two_surface_DP).parameters.keys())

    def _one(params):
        call_kwargs = {k: v for k, v in params.items() if k in allowed}
        y1, y2, dbg = run_two_surface_DP(return_debug=True, **call_kwargs)
        # return {"params": params, "call_kwargs": call_kwargs, "y1": y1, "y2": y2, "debug": dbg}
        return {"params": params,  "y1": y1, "y2": y2, "debug": dbg}

    return Parallel(n_jobs=n_jobs)(delayed(_one)(p) for p in param_list)
# def sweep_two_surface_dp(
#     # cost1=None, cost2=None,
#     param_list,
#     n_jobs: int = 8,
# ):
#     """
#     param_list: list of dicts passed into run_two_surface_DP (must include core params)
#     Returns list of dicts: { 'params':..., 'y1':..., 'y2':..., 'debug':... }
#     """
#     import inspect


#     def _one(params):
#         # y1, y2, dbg = run_two_surface_DP(cost1, cost2, return_debug=True, **params)
#         y1, y2, dbg = run_two_surface_DP(return_debug=True, **params)
#         return {"params": params, "y1": y1, "y2": y2, "debug": dbg}

#     return Parallel(n_jobs=n_jobs)(delayed(_one)(p) for p in param_list)


def add_dp_sweep_to_arrayboard(
    AB,
    img_band,
    img_full,
    band_top: int,
    results,
    title_prefix: str = "",
    max_overlays: int = 8,
    offset_value = None,
    img_band_radius = 30,
    relevant_names = ["lambda2","single_kappa",'darkness_barrier_factor'],
):
    """
    Adds:
      - band image with multiple y1/y2 overlays
      - full image with shifted overlays
      - 1D plots: separation d, dist-to-anchor2, gap/kappa
    """
    from copy import deepcopy

    # overlay only first N results to avoid spaghetti
    # results = results[:max_overlays]

    # # --- band overlays
    # lines_band = {}
    # for k, r in enumerate(results):
    #     tag = f"{k}"
    #     y1 = r["y1"]
    #     y2 = r["y2"]
    #     lines_band[f"y1_{tag}"] = y1.copy()
    #     lines_band[f"y2_{tag}"] = y2.copy()

    # AB.add(img_band, lines=lines_band, title=fill(f"{title_prefix} band overlays",width=30))

    # # --- full overlays (shift by band_top)
    # lines_full = {}
    # for k, r in enumerate(results):
    #     tag = f"{k}"
    #     y1 = r["y1"] + band_top
    #     y2 = r["y2"] + band_top
    #     lines_full[f"y1_{tag}"] = y1
    #     lines_full[f"y2_{tag}"] = y2

    # AB.add(img_full, lines=lines_full, title=fill(f"{title_prefix} full overlays",width=30))

    # --- 1D plots per result (compact but useful)
    from scipy.ndimage import median_filter,uniform_filter1d
    import matplotlib.pyplot as plt
    def _plot_dict(ax, plot_dict, legend=True):
        cycle_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for i, (name, v) in enumerate(plot_dict.items()):
            c = cycle_colors[i % len(cycle_colors)]

            ax.plot(v, linewidth=0.5, alpha=0.4, label=name, color=c)

            v_med = median_filter(v, size=20, mode="nearest")
            ax.plot(v_med, linewidth=0.9, alpha=0.8, label=f"{name} (med)", color=c)

        if legend:
            ax.legend(fontsize=6, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))




    for k, r in enumerate(results):
        lines_band = {}
        tag = f"{k}"
        y1 = r["y1"]
        y2 = r["y2"]
        lines_band[f"y1_{tag}"] = y1.copy()
        lines_band[f"y2_{tag}"] = y2.copy()


        # params_string = " ".join([f'{n}:{r['params'][n]}' for n in relevant_names])
        filtered_names = [n for n in relevant_names if n in r['params']]
        params_string = " ".join([f"{n}:{r['params'].get(n, 'NA')}" for n in filtered_names])

        if 'cost2' in r['params']:
            img_to_plot = 1-r['params']['cost2']
        else:
            img_to_plot = img_band
        AB.add(img_to_plot, lines=lines_band, title=fill(f"{title_prefix} band overlays {params_string}",width=30))




        # dbg = r['dbg']
        dbg = r["debug"]
        parts = None
        if "peaks" in dbg:
            print("adding peaks")
            peaks = dbg['peaks']
            peak_img = spu.overlay_peaks_on_image(img_to_plot,peaks)
            AB.add(peak_img,  title='peak img')

            # gof,parts = suf.two_line_gof_from_peaks_and_gap(img_band,y1-img_band_radius+img_band_radius,y2-img_band_radius+offset_value,peaks=peaks,dmin=10,return_parts=True,peak_tol=3)
            gof,parts = suf.two_line_gof_from_peaks_and_gap(img_band,y1,y2,peaks=peaks,dmin=10,return_parts=True,peak_tol=3)
        else:
            print("skipping peak img")

        if 'barrier' in dbg:
            AB.add(dbg['barrier'],  title='barrier img')


        print("now processing the output lines")



        met = dp_two_surface_debug_metrics(
            cost1=None if False else None,  # unused here
            cost2=None if False else None,  # unused here
            y1=r["y1"], y2=r["y2"], debug=dbg
        )
        # NOTE: we want anchor distance etc using cost2; but met above expects cost2.
        # We’ll instead plot what we already have from dbg (gap2, kappa) + d:
        plot_dict = {
            # "d=y2-y1": (r["y2"] - r["y1"]).astype(float),
        }
        if "gap2" in dbg:
            plot_dict["gap2(second-best - best)"] = np.asarray(dbg["gap2"], float)
        if "kappa" in dbg:
            plot_dict["kappa(anchor strength)"] = np.asarray(dbg["kappa"], float)

        if parts:
            for part_name in [ 'intensity1', 'intensity2', 'intensity_score']:
                plot_dict[part_name] = parts[part_name]
 
        # relevant_names = ["lambda2","single_kappa"]
        # params_string = " ".join([f'{n}:{r['params'][n]}' for n in relevant_names])
        params_string = " ".join([f"{n}:{r['params'].get(n, 'NA')}" for n in relevant_names])


        if False:
            AB.add_plot(lambda ax, pd=deepcopy(plot_dict): _plot_dict(ax, pd),
                        title=fill(f"{title_prefix} metrics run {k} params={params_string}",width=30))


# def plot_parts_norm(ax,parts,filter_type,norm=True):
#     for k,v in parts.items():
#         if len(v)!=512:
#             continue
#         if norm:
#             v_norm = v/np.max(v)
#         else:
#             v_norm = v
#         if filter_type == 'mean':
#             y_med = uniform_filter1d(v_norm, size=50, mode="nearest")
#         elif filter_type == 'median':
#             y_med = median_filter(v_norm, size=25, mode="nearest")
#             print("using median filter")
#         else:
#             raise Exception("must supply filter type")
#         ax.plot(y_med,alpha=0.5,linewidth=0.8,label=f"{k}, range=[{round(np.min(v),2)},{round(np.max(v),2)}]")
#     ax.legend(fontsize=4, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))

