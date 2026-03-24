from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))  # adds Han_AIR/ to path
sys.path.append(str(Path(__file__).resolve().parents[2]))  # adds Han_AIR/ to path
import numpy as np
from scipy import ndimage
import code_files.visualization_utils as vu
import code_files.file_utils as fu
import code_files.zarr_file_utils as zfu
from dask import array as da
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
from functools import wraps

def get_centroids_from_annotations(zarr_path, onh_label=1, fovea_label=2):
    """
    Compute the (y, x) centroid positions for ONH (ellipse) and fovea (circle)
    from a 3D annotation Zarr volume.

    Parameters
    ----------
    zarr_path : str or Path
        Path to the Zarr annotation volume.
    onh_label : int
        Label value corresponding to the ONH region (ellipse).
    fovea_label : int
        Label value corresponding to the fovea region (circle).

    Returns
    -------
    dict
        {'onh_center': (y, x), 'fovea_center': (y, x)}
    """

    # Load the zarr array (can be large; no need to compute everything)
    # z = zarr.open(zarr_path, mode='r')
    z = da.from_zarr(zarr_path)

    # We assume z has shape (Z, Y, X)
    # Collapse along Z to get en-face projections
    proj = z.max(axis=1)  # en-face label map (Y, X)
    print(f"proj.shape is {proj.shape}")
    

    # Get binary masks for each region
    onh_mask = (proj == onh_label)
    fovea_mask = (proj == fovea_label)

    # Compute centroids if present
    def _centroid(mask):
        if mask.sum() == 0:
            return None
        coords = ndimage.center_of_mass(mask)
        return tuple(float(c) for c in coords)  # (y, x)

    onh_center = _centroid(onh_mask)
    fovea_center = _centroid(fovea_mask)

    return {'onh_center': onh_center, 'fovea_center': fovea_center}

def skip_if_exists(key_fields=(),recompute = lambda: GLOBAL_RECOMPUTE,all_existing_maps = lambda: ALL_EXISTING_MAPS):
    """Decorator: return (key, None) if key exists and RECOMPUTE is False.
    Key = fn.__name__ + '|' + '|'.join(str(kwargs[f]) for f in key_fields)"""
    def deco(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            key_parts = [fn.__name__] + [str(kwargs[f]) for f in key_fields]
            key = "|".join(key_parts)
            if (not recompute) and (key in all_existing_maps):
                return key, None
            out = fn(*args, **kwargs)
            return key, out
        return wrapper
    return deco

class mapGenerators(object):
    """very importatnt to only assign the wrapper skip_if_exists to true static method output functions"""
    @staticmethod
    def _yzx_dims(image):
        """Validate that image is (Z, Y, X) and return Z,Y,X."""
        if image.ndim != 3:
            raise ValueError("image must be 3D (Z, Y, X)")
        Z, Y, X = image.shape
        return Z, Y, X

    @staticmethod
    def _yz_broadcast_coords(Y, Z=None, X=None):
        """
        Return y_coords as shape (1, Y, 1) for broadcasting.
        Z, X are unused but kept for readability where called.
        """
        return np.arange(Y, dtype=np.int32)[None, :, None]

    @staticmethod
    @skip_if_exists()
    def thickness_map(layers):
        """Return per-(Z,X) thickness = ILM - RPE."""
        return layers['ilm_smooth'] - layers['rpe_smooth']

    @staticmethod
    def grabslab(image, single_layer, y_range):
        """
        Grab a constant-thickness slab around a single constant layer height.

        Parameters
        ----------
        image : (Z,Y,X)
        single_layer : scalar-like layer height (same for all (Z,X))
        y_range : [y_min_offset, y_max_offset]. Example: [-10, +10]
                  Inclusive of endpoints in intent; slicing uses [y1:y0].
        """
        assert np.ndim(single_layer) == 0 or (np.size(np.unique(single_layer)) == 1)
        layer_height = int(np.array(single_layer).reshape(-1)[0])

        # Convert offsets (relative to layer) into absolute Y indices.
        # Example: y_range=[-10, +10] -> y from (layer-10) to (layer+10)
        y0 = layer_height - y_range[0]
        y1 = layer_height - y_range[1]

        # Ensure proper order for slicing
        y_lo, y_hi = (y1, y0) if y1 < y0 else (y0, y1)

        # Clip to image bounds
        Z, Y, X = mapGenerators._yzx_dims(image)
        y_lo = max(0, min(Y, y_lo))
        y_hi = max(0, min(Y, y_hi))

        return image[:, y_lo:y_hi, :]

    @staticmethod
    @skip_if_exists(key_fields=['y_range'])
    def basic_slab_avg_map(image, single_layer, y_range):
        """
        Average intensity over a constant-thickness slab around a constant layer.
        Returns a (Z, X) map.
        """
        slab = mapGenerators.grabslab(image, single_layer, y_range)  # (Z, Yslab, X)
        return np.mean(slab, axis=1)  # -> (Z, X)

    # ---------- general masks for curved surfaces ----------
    @staticmethod
    def _mean_between_layers(image, lower_y, upper_y):
        """
        Average per (Z,X) between two Y surfaces given as integer arrays of shape (Z, X).
        Order is automatically handled (we don't assume which is larger).
        Out-of-range handled by clipping. Uses np.nanmean (robust to empty).
        """
        Z, Y, X = mapGenerators._yzx_dims(image)

        if lower_y.shape != (Z, X) or upper_y.shape != (Z, X):
            raise ValueError("lower_y and upper_y must be shape (Z, X)")

        y_coords = mapGenerators._yz_broadcast_coords(Y, Z, X)  # (1, Y, 1)
        # Expand to (Z, 1, X) for broadcasting
        y_lo = np.minimum(lower_y, upper_y)[:, None, :]
        y_hi = np.maximum(lower_y, upper_y)[:, None, :]
        # Clamp to [0, Y-1]
        y_lo = np.clip(y_lo, 0, Y - 1)
        y_hi = np.clip(y_hi, 0, Y - 1)

        mask = (y_coords >= y_lo) & (y_coords <= y_hi)  # (Z, Y, X)
        # Avoid integer overflow if image is integer; cast to float for NaNs
        imgf = image.astype(np.float32, copy=False)
        masked = np.where(mask, imgf, np.nan)
        return np.nanmean(masked, axis=1)  # (Z, X)

    @staticmethod
    def _mean_above_layer(image, layer_y):
        """
        Average per (Z,X) for all Y strictly above 'layer_y' (vitreous side).
        Assumes Y=0 is vitreous and increasing Y goes deeper.
        """
        Z, Y, X = mapGenerators._yzx_dims(image)
        if layer_y.shape != (Z, X):
            raise ValueError("layer_y must be shape (Z, X)")

        y_coords = mapGenerators._yz_broadcast_coords(Y, Z, X)  # (1, Y, 1)
        y_top = np.clip(layer_y, 0, Y)[:, None, :]  # (Z,1,X). Above means y < layer
        mask = (y_coords < y_top)
        imgf = image.astype(np.float32, copy=False)
        masked = np.where(mask, imgf, np.nan)
        return np.nanmean(masked, axis=1)  # (Z, X)

    @staticmethod
    def _mean_below_layer(image, layer_y):
        """
        Average per (Z,X) for all Y strictly below 'layer_y' (choroid side).
        """
        Z, Y, X = mapGenerators._yzx_dims(image)
        if layer_y.shape != (Z, X):
            raise ValueError("layer_y must be shape (Z, X)")

        y_coords = mapGenerators._yz_broadcast_coords(Y, Z, X)  # (1, Y, 1)
        y_bot = np.clip(layer_y, -1, Y - 1)[:, None, :]  # below means y > layer
        mask = (y_coords > y_bot)
        imgf = image.astype(np.float32, copy=False)
        masked = np.where(mask, imgf, np.nan)
        return np.nanmean(masked, axis=1)  # (Z, X)

    # ---------- your requested maps ----------
    @staticmethod
    @skip_if_exists(key_fields=())
    def full_retina_avg_map(image, layers):
        """
        Mean reflectivity between ILM and RPE surfaces (inclusive).
        Works regardless of whether you've flattened.
        Returns (Z, X).
        """
        ilm = layers['ilm_smooth'].astype(np.int32)
        rpe = layers['rpe_smooth'].astype(np.int32)
        return mapGenerators._mean_between_layers(image, rpe, ilm)

    @staticmethod
    @skip_if_exists(key_fields=())
    def vitreous_avg_map(image, layers):
        """
        Mean reflectivity above ILM (vitreous side).
        Does NOT flatten first; uses a curved mask & np.nanmean.
        Returns (Z, X).
        """
        ilm = layers['ilm_smooth'].astype(np.int32)
        return mapGenerators._mean_above_layer(image, ilm)

    @staticmethod
    @skip_if_exists(key_fields=())
    def choroid_avg_map(image, layers):
        """
        Mean reflectivity below RPE (choroid side).
        If you've already flattened to RPE, this still works.
        Returns (Z, X).
        """
        rpe = layers['rpe_smooth'].astype(np.int32)
        return mapGenerators._mean_below_layer(image, rpe)

    @staticmethod
    @skip_if_exists(key_fields=['y_range'])
    def basic_slab_variance_map(image, single_layer, y_range, ddof=0):
        """
        Variance across Y within a constant-thickness slab around a constant layer.
        Returns (Z, X).
        """
        slab = mapGenerators.grabslab(image, single_layer, y_range)  # (Z, Yslab, X)
        return np.var(slab.astype(np.float32, copy=False), axis=1, ddof=ddof)

    def _detrend_median(slab):
        """Quick robust detrend per column (Z,X) across Y: subtract column-wise median."""
        med = np.nanmedian(slab, axis=1, keepdims=True)  # (Z,1,X)
        return slab - med

    def _mad_first_diff(slab):
        """Median Absolute Deviation of first differences along Y -> (Z,X)."""
        d = np.diff(slab, axis=1)  # (Z, Y-1, X)
        mad = np.nanmedian(np.abs(d - np.nanmedian(d, axis=1, keepdims=True)), axis=1)
        return mad.astype(np.float32)








def get_flat_img_and_flat_label(vp,dir_suffix):
    """vp is a volume path. from this return da.from_zarr(str(p)) teh string for the zarr to the flat img and 
    the npzfile object for the layers flattened according to same structure (rpe smooth)"""
    flattener = 'rpe_smooth'
    vol_flat_vp = zfu.ensure_image_flat_zarr(vp,flatten_with=flattener,z_stride=1,overwrite=LAYERS_OVERWRITE,dir_suffix=dir_suffix) # The tricky thing is you can't just reload the old flattened vol. need to overwrite if you wnat recomputation with new layers
    vol_flat = da.from_zarr(str(vol_flat_vp))     # (Z,Y,X)

    # layers_obj = np.load(fu.get_corresponding_layer_path(vp, file_suffix='_layers.npz', dir_suffix=dir_suffix))
    layers_obj = np.load(fu.get_corresponding_layer_path(vp, file_suffix='_layers.npz', dir_suffix=dir_suffix))
    # import pdb; pdb.set_trace()
    flat_layers = vu.flatten_other_curves(layers_obj, ref_layer=flattener, output_height=int(vol_flat.shape[1])) # 
    return vol_flat,flat_layers

# def get_maps(map_function_list):
#     """get all maps in the map function list. A list of static methods"""
#     output_maps = {}
#     for map_fn in map_function_list:
#         output_maps[map_fn.__name__] = map_fn

def get_maps(image,layers):
    """Keep it simple"""
    output = {}
    
    key,output[key] = mapGenerators.thickness_map(layers)
    thincut_ranges = [[i,i+5] for i in range(0,40,5)]
    all_ranges = [[0,20],[20,60],[5,30],[5,25]] + thincut_ranges # suspecting 5-25 is the ideal extraction range. This should funnel into downstream extractions
    for y_range in all_ranges:
        key,output[key] = mapGenerators.basic_slab_avg_map(image,layers['rpe_smooth'],y_range=y_range)
        key,output[key] = mapGenerators.basic_slab_variance_map(image, layers['rpe_smooth'],y_range=y_range)
    key,output[key] = mapGenerators.full_retina_avg_map(image, layers)
    key,output[key] = mapGenerators.vitreous_avg_map(image, layers)
    key,output[key] = mapGenerators.choroid_avg_map(image, layers)
    return output

def _parallel_process_map(vp,layer_dir_suffix):
    name = vp.stem
    vol_flat, layers_obj = get_flat_img_and_flat_label(vp, layer_dir_suffix)
    dict_entry = {'path':str(vp),'vol':vol_flat,'layers':layers_obj,'maps':{}}

    updater = get_maps(vol_flat,layers_obj)
    dict_entry['maps'].update(updater)
    return name,dict_entry 
    # maps_dict[name]['thickness_map'] = mapGenerators.thickness_map(maps_dict[name]['layers'])


def get_all_maps(root_dir,layer_dir_suffix='_layers_08_25_25',glob='*.img',parallel=False):
    """lets use the zarr format here s.t. we don't overload ram. We will call the zarr flat functions to use labels 
    flattened according to the RPE to simplify any slab calculation functions. 
    This should be able to run very quickly, as we are not going to be loading a ton of things at once into ram.
    likely we can work wtih certain slices given the flattening used

    We will load the images and calculate their feature maps, then return them
    
    inputs: root_dir -- directory where the images all live (will be the original data directory). 
    layer_dir_suffix: should be the most-recntly calculated layers. currently 8/25/25, but #TODO will re-reun these w/ the new annotation"""
    if not parallel:
        print('processing single maps')
        maps_dict = {}

        ALL_VOL_PATHS = sorted(Path(root_dir).glob(glob))
        for vp in tqdm(ALL_VOL_PATHS):
            name = vp.stem
            vol_flat,layers_obj = get_flat_img_and_flat_label(vp,layer_dir_suffix)
            maps_dict[name] = {'path':vp,'vol':vol_flat,'layers':layers_obj,'maps':{}}

            updater = get_maps(vol_flat,layers_obj)
            maps_dict[name]['maps'].update(updater)
            maps_dict[name]['thickness_map'] = mapGenerators.thickness_map(maps_dict[name]['layers'])


    if parallel:
        print("processing maps in parallel")

        maps_dict = {}
        ALL_VOL_PATHS = sorted(Path(root_dir).glob(glob))
        # then compute maps in parallel
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=GLOBAL_MAX_WORKERS) as ex:
            futures = [
                ex.submit(_parallel_process_map, vp,layer_dir_suffix)
                for vp in ALL_VOL_PATHS
            ]
            for fut in tqdm(futures, desc="Computing maps"):
                name,dict_entry = fut.result()
                maps_dict[name] = dict_entry

    return maps_dict

    

def update_all_maps_dict():
    """iterates thru a loaded maps dict and updates with any that were computed on this run"""
    pass

def plot_maps(maps_dict,key):
    ncols = 4
    nrows = math.ceil(len(maps_dict)/ncols)
    fig,axes = plt.subplots(nrows,ncols,dpi=400)
    axes = axes.flatten()

    for i,name in enumerate(maps_dict.keys()):
        axes[i].imshow(maps_dict[name]['maps'][key],cmap='gray',aspect='auto')
    for ax in axes:
        ax.axis('off')

    fig.suptitle(key)
    plt.tight_layout()
    plt.savefig(MAP_OUTPUT_DIR / f"{key}.png")
    plt.close()

def quickfig(array):
    plt.figure()
    plt.imshow(array,cmap='gray')
    plt.show()





import numpy as np
from skimage.transform import AffineTransform, warp, resize
from skimage.feature.texture import graycomatrix, graycoprops

# ---------------------------
# 1) Align by fovea–ONH line
# ---------------------------

def align_by_fovea_onh(map_zx, fovea_yx, onh_yx, target_angle_deg=0.0, output_shape=None, order=1, cval=np.nan):
    """
    Rigidly align a (Z,X) en-face map so that:
      - ONH is moved to the origin (0,0) in the transformed coordinates,
      - the ONH→fovea line is rotated to 'target_angle_deg' (default: 0°, i.e., horizontal to +X).

    Parameters
    ----------
    map_zx : 2D array (Z, X)
    fovea_yx, onh_yx : tuples (z, x) in the same coordinate system as map_zx
                       (Use the en-face centers you computed; treat row->Z, col->X.)
    target_angle_deg : float
        Desired angle (degrees) of the ONH→fovea vector after rotation, measured from +X toward +Z.
    output_shape : (Z_out, X_out) or None
        If None, uses input shape.
    order : int
        Interpolation order for warp (0=nearest, 1=bilinear, 3=bicubic).
    cval : float
        Fill value for areas outside input after warping.

    Returns
    -------
    aligned : 2D array (Z_out, X_out)
    M_2x3 : np.ndarray shape (2,3)
        The forward affine matrix applied by skimage (mapping output coords -> input coords).
    """
    Z, X = map_zx.shape
    if output_shape is None:
        output_shape = (Z, X)

    # Vector from ONH to fovea in (Z,X)
    dz = float(fovea_yx[0] - onh_yx[0])
    dx = float(fovea_yx[1] - onh_yx[1])

    # Angle of this vector relative to +X (radians), accounting for row-major image coords
    theta = np.arctan2(dz, dx)  # because rows (Z) increase downward
    target_theta = np.deg2rad(target_angle_deg)
    dtheta = target_theta - theta  # rotate by this amount

    # Compose: translate(-ONH) -> rotate(dtheta) -> (optional) translate to keep content roughly centered
    t1 = AffineTransform(translation=(-onh_yx[1], -onh_yx[0]))  # (x,y) order inside transform
    r  = AffineTransform(rotation=dtheta)

    # Optional: center the ONH near output origin or center; here keep ONH at 0,0 in output.
    t2 = AffineTransform(translation=(0.0, 0.0))

    A = (t1 + r)  # note: AffineTransform supports addition as composition (t2 @ r @ t1 would be more explicit)
    A = AffineTransform(matrix=(t2.params @ r.params @ t1.params))  # ensure full composition in right order

    # skimage.warp uses "inverse_map": output -> input mapping
    aligned = warp(
        map_zx.astype(np.float32, copy=False),
        inverse_map=A.inverse,
        output_shape=output_shape,
        order=order,
        preserve_range=True,
        cval=cval,
    ).astype(np.float32, copy=False)

    # Return a 2x3 forward matrix (for convenience) matching OpenCV-style
    M_3x3 = A.params  # 3x3
    M_2x3 = M_3x3[:2, :]
    return aligned, M_2x3


# -------------------------------------------
# 2) GLCM feature maps over tiled (Z,X) image
# -------------------------------------------

def glcm_feature_maps(
    map_zx,
    window=(32, 32),
    step=(32, 32),
    num_levels=32,
    clip_percentiles=(1, 99),
    angles_deg=(0, 90, 45, -45),
    distances=(1,),
    features=("contrast", "correlation", "homogeneity", "ASM", "energy", "entropy"),
):
    """
    Compute tiled GLCM features on a (Z,X) map and average across angles (rotation-robust).

    Steps:
      1) Robustly quantize map to integer levels [0..num_levels-1]
      2) Slide/stride windows of size 'window' every 'step'
      3) For each tile, compute GLCM at given distances/angles
      4) Average per-feature across angles (and distances if >1)
      5) Return a dict of feature maps with shape (n_tiles_z, n_tiles_x)

    Parameters
    ----------
    map_zx : 2D array (Z, X)
    window : (wz, wx)
    step   : (sz, sx)
    num_levels : int
    clip_percentiles : (low, high) for robust intensity clipping before quantization
    angles_deg : iterable of angles in degrees
    distances : iterable of distances (integers)
    features : iterable of feature names; supports: contrast, correlation, homogeneity, ASM, energy, entropy

    Returns
    -------
    feats : dict[str, 2D np.ndarray]
        Each has shape (n_tiles_z, n_tiles_x)
    meta : dict
        Includes 'grid_shape', 'window', 'step', 'levels', 'angles', 'distances'
    """
    z, x = map_zx.shape
    wz, wx = window
    sz, sx = step

    # Compute tile grid
    idx_z = list(range(0, max(1, z - wz + 1), sz))
    idx_x = list(range(0, max(1, x - wx + 1), sx))
    nZ, nX = len(idx_z), len(idx_x)

    # Robust quantization to levels 0..L-1
    m = map_zx.astype(np.float32, copy=False).compute() # move from dask array to numpy
    lo, hi = np.nanpercentile(m, clip_percentiles)
    m_clipped = np.clip(m, lo, hi)
    # Avoid division by zero if lo==hi
    if hi > lo:
        m_norm = (m_clipped - lo) / (hi - lo)
    else:
        m_norm = np.zeros_like(m_clipped, dtype=np.float32)
    q = np.floor(m_norm * (num_levels - 1) + 0.5).astype(np.uint8)

    # Angles in radians
    angles = np.deg2rad(np.array(angles_deg, dtype=float))

    # Storage
    out = {f: np.full((nZ, nX), np.nan, dtype=np.float32) for f in features}

    # Iterate tiles
    for i, z0 in enumerate(idx_z):
        for j, x0 in enumerate(idx_x):
            tile = q[z0:z0 + wz, x0:x0 + wx]
            if tile.size == 0:
                continue

            # GLCM: shape (levels, levels, num_distances, num_angles)
            glcm = graycomatrix(
                tile,
                distances=distances,
                angles=angles,
                levels=num_levels,
                symmetric=True,
                normed=True,
            )

            # Built-ins first
            for fname in features:
                if fname in ("contrast", "correlation", "homogeneity", "ASM", "energy"):
                    vals = graycoprops(glcm, fname)  # shape (num_distances, num_angles)
                    out[fname][i, j] = np.nanmean(vals)
                elif fname == "entropy":
                    # Manual entropy over all (d, a)
                    # glcm: (L,L,D,A)
                    P = glcm.astype(np.float64, copy=False)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        ent = -(P * np.log2(P + 1e-12))
                    # mean over (d,a), sum over (i,j)
                    ent_da = ent.sum(axis=(0, 1))  # -> (D, A)
                    out["entropy"][i, j] = float(np.nanmean(ent_da))
                else:
                    raise ValueError(f"Unknown GLCM feature: {fname}")
        

    meta = dict(
        grid_shape=(nZ, nX),
        window=window,
        step=step,
        levels=num_levels,
        angles_deg=list(angles_deg),
        distances=list(distances),
        clip_percentiles=clip_percentiles,
    )
    return out, meta


# ------------------------------------------
# 3) OD/OS alignment + asymmetry computation
# ------------------------------------------

def od_os_asymmetry(
    od_map,
    os_map,
    od_fovea_yx,
    od_onh_yx,
    os_fovea_yx,
    os_onh_yx,
    canonical_angle_deg=0.0,
    out_shape=None,
    flip_os=True,
    order=1,
    cval=np.nan,
    roi_masks=None,
    summary_percentiles=(50, 75, 90),
):
    """
    Align OD and OS maps to a common right-eye canonical frame, optionally mirror OS,
    and compute |OD - OS| asymmetry + robust summaries.

    Parameters
    ----------
    od_map, os_map : 2D arrays (Z, X)
    *_fovea_yx, *_onh_yx : tuples (z, x) for each eye
    canonical_angle_deg : float
        Rotate each eye so ONH->fovea aligns to this angle (0° = horizontal to +X)
    out_shape : (Z_out, X_out) or None
        If None, uses max of both shapes after alignment to avoid cropping.
    flip_os : bool
        If True, mirror OS horizontally into a right-eye frame (along X axis).
    order : int
        Interpolation order used during warps.
    cval : float
        Fill value for outside regions.
    roi_masks : dict[str, 2D np.ndarray] or None
        Optional masks (same shape as output) to summarize asymmetry by region,
        e.g., {'macula': mask1, 'peripap': mask2}
    summary_percentiles : tuple
        Percentiles to report.

    Returns
    -------
    result : dict
        {
          'od_aligned': 2D,
          'os_aligned': 2D,
          'os_aligned_flipped': 2D,
          'asym_map': 2D,
          'summaries': {
              'global': {'p50':..., 'p75':..., ...},
              'macula': {...}, ...
          }
        }
    """
    # 1) Align each eye by its landmarks to the canonical angle
    od_aligned, _ = align_by_fovea_onh(
        od_map, od_fovea_yx, od_onh_yx,
        target_angle_deg=canonical_angle_deg,
        output_shape=od_map.shape if out_shape is None else out_shape,
        order=order, cval=cval
    )
    os_aligned, _ = align_by_fovea_onh(
        os_map, os_fovea_yx, os_onh_yx,
        target_angle_deg=canonical_angle_deg,
        output_shape=os_map.shape if out_shape is None else out_shape,
        order=order, cval=cval
    )

    # 2) Choose a common output shape
    if out_shape is None:
        Z_out = max(od_aligned.shape[0], os_aligned.shape[0])
        X_out = max(od_aligned.shape[1], os_aligned.shape[1])
        out_shape = (Z_out, X_out)

    def _to_shape(img, shape):
        if img.shape == shape:
            return img
        return resize(img, shape, order=order, preserve_range=True, anti_aliasing=False).astype(np.float32)

    od_aligned = _to_shape(od_aligned, out_shape)
    os_aligned = _to_shape(os_aligned, out_shape)

    # 3) Mirror OS horizontally (X axis) into a right-eye frame if requested
    os_aligned_flipped = os_aligned[:, ::-1] if flip_os else os_aligned

    # 4) Asymmetry
    asym = np.abs(od_aligned - os_aligned_flipped).astype(np.float32)

    # 5) Summaries (global + optional ROIs)
    def _perc(a, mask=None):
        v = a if mask is None else a[mask.astype(bool)]
        v = v[np.isfinite(v)]
        if v.size == 0:
            return {f"p{p}": np.nan for p in summary_percentiles}
        return {f"p{p}": float(np.percentile(v, p)) for p in summary_percentiles}

    summaries = {"global": _perc(asym)}
    if roi_masks is not None:
        for name, m in roi_masks.items():
            if m.shape != asym.shape:
                raise ValueError(f"ROI mask '{name}' has shape {m.shape}, expected {asym.shape}")
            summaries[name] = _perc(asym, mask=m)

    return dict(
        od_aligned=od_aligned,
        os_aligned=os_aligned,
        os_aligned_flipped=os_aligned_flipped,
        asym_map=asym,
        summaries=summaries,
    )
###################


    
GLOBAL_RECOMPUTE = True
LAYERS_OVERWRITE = True
GLOBAL_MAX_WORKERS=6
if __name__ == '__main__':
    MAP_OUTPUT_DIR = Path("/Users/matthewhunt/Research/Iowa_Research/Han_AIR/reports/enface_analysis_outputs/")
    glob = '*.png'
    ALL_EXISTING_MAPS = {p.stem for p in Path(MAP_OUTPUT_DIR).glob(glob)}
    print(f"ALL_EXISTING_MAPS is {ALL_EXISTING_MAPS}")

    import pickle
    root_dir = 'data_all_volumes'
    maps_dict = get_all_maps(root_dir,layer_dir_suffix="_layers_2025-11-19",parallel=True)
    pickle.dump(maps_dict,open('/Users/matthewhunt/Research/Iowa_Research/Han_AIR/results/all_maps_dict.pickle','wb'))
    all_map_keys = list(maps_dict.values())[0]['maps'].keys()
    for k in all_map_keys:
        if GLOBAL_RECOMPUTE or k not in ALL_EXISTING_MAPS:
            plot_maps(maps_dict,k)

