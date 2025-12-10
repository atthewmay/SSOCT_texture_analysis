import os
import numpy as np
import matplotlib.pyplot as plt

# ───── CONFIG ──────────────────────────────────────────────────────────────

SHAPE     = (1024, 1536, 512)             # (Z, height, width)
DTYPE     = np.uint16                     # raw dtype
N_SLICES  = 7                             # how many slices per eye
STRIDE    = 10                            # spacing between slices
ASPECT    = 1/7                           # matplotlib aspect ratio (pixel x/y)

# ───── FUNCTION ────────────────────────────────────────────────────────────

def plot_center_slices(base_dir):
    for category in sorted(os.listdir(base_dir)):
        cat_path = os.path.join(base_dir, category)
        if not os.path.isdir(cat_path):
            continue

        for patient in sorted(os.listdir(cat_path)):
            pat_path = os.path.join(cat_path, patient)
            if not os.path.isdir(pat_path):
                continue

            # find OD & OS cube_z.img
            paths = {"OD": None, "OS": None}
            enface_paths = {"OD": None, "OS": None}
            for f in os.listdir(pat_path):
                if f.endswith("cube_z.img"):
                    if "_OD_" in f:
                        paths["OD"] = os.path.join(pat_path, f)
                    elif "_OS_" in f:
                        paths["OS"] = os.path.join(pat_path, f)

                if f.endswith("lslo.bin"):
                    if "_OD_" in f:
                        enface_paths["OD"] = os.path.join(pat_path, f)
                    elif "_OS_" in f:
                        enface_paths["OS"] = os.path.join(pat_path, f)


            # skip if missing either eye
            if not all(paths.values()):
                continue

            # load volumes
            vols = {}
            for eye, fp in paths.items():
                flat = np.fromfile(fp, dtype=DTYPE)
                vols[eye] = flat.reshape(SHAPE)



            # ───── INSERT: load & plot overview enface ────────────────────────────
            if enface_paths:
                enface_shape = (512, 332)
                fig_enf, axs_enf = plt.subplots(1, 2, figsize=(8, 4))
                for idx_eye, eye in enumerate(["OD", "OS"]):
                    # build filename and load
                    enf_fp = enface_paths[eye]
                    enf_flat = np.fromfile(enf_fp, dtype=DTYPE)
                    enf = enf_flat.reshape(enface_shape)
                    # rotate 90° into examiner’s perspective
                    enf = np.rot90(enf,k=-1)
                    # plot
                    ax = axs_enf[idx_eye]
                    ax.imshow(enf, cmap="gray", aspect="equal")
                    ax.set_title(f"{patient} {eye} enface")
                    ax.axis("off")
                plt.tight_layout()
                plt.show()
                
            # compute slice indices
            Z = SHAPE[0]
            mid = Z // 2
            start = max(0, mid - (N_SLICES//2)*STRIDE)
            idxs  = list(range(start, start + N_SLICES*STRIDE, STRIDE))

            # plot OD & OS
            fig, axes = plt.subplots(
                2, len(idxs),
                figsize=(4*len(idxs), 8),
                squeeze=False
            )

            for row, eye in enumerate(["OD", "OS"]):
                vol = vols[eye]
                for col, z in enumerate(idxs):
                    img = np.rot90(vol[z],k=2)
                    ax  = axes[row, col]
                    ax.imshow(img, cmap="gray", aspect=ASPECT)
                    ax.set_title(f"{eye} slice {z}")
                    ax.axis("off")

            plt.suptitle(f"{patient}  [{category}]", y=0.95)
            plt.tight_layout()
            plt.show()

import numpy as np
import matplotlib.pyplot as plt
import random

def plot_slices_with_layers(volume,layers=None, indices = range(10,20), max_cols = 7, layer_names=None,return_fig = False):
    """
    # Load the prediction array
    volume = np.load(npy_path)  # shape: (n_slices, H, W, n_layers)
    """
    import math
    _, H, W = volume.shape
    print(f"image of shape {W} wide by {H} tall")


    # Plot
    ncols = min(max_cols,len(indices))
    nrows = math.ceil(len(indices)/ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    # axs = np.atleast_2d(axs)

    if len(indices) == 1:
        axs = [axs]

    for ax, idx in zip(axs, indices):
        # ax.imshow(np.arange(H).reshape(-1, 1), cmap="gray", aspect='auto', extent=[0, W, H, 0], alpha=0)  # for axis formatting
        ax.imshow(volume[idx, :, :], cmap="gray", aspect='auto')  # use first layer as background if desired

        if layers is not None:
            for layer_idx in range(layers.shape[-1]):
                # line = layers[idx, :, layer_idx]  # flip it
                line = layers[idx, :, layer_idx]  # flip it
                # line = line/H # normalize
                ax.plot(np.arange(W), line, label=layer_names[layer_idx] if layer_names else f"Layer {layer_idx}")
                # For a boundary, we assume it's a height map: one row per column
                # line = height_map.mean(axis=0)  # just in case — collapse any residual H
        ax.set_title(f"Slice {idx}")
        ax.set_axis_off()
        ax.legend()

    if return_fig:
        return fig
    plt.tight_layout()
    plt.show()


#REFACTOR: put this as a method of the PlotTracer
def plot_all_traces(all_traces_list,suptitle = None,dpi=150):
    """will be a list of lists, where each sublist is from a single slice, processed in order, thus these should be laid along a column"""
    nrows = len(all_traces_list[0]) #rows are bscan slices
    ncols = len(all_traces_list) #rows are bscan slices
    figsize = (12,8)
    fig,axes = plt.subplots(nrows,ncols,figsize = (figsize[0], figsize[1]*nrows/ncols),dpi=dpi)


    for slice_idx in range(ncols):
        for seq_idx in range(nrows):
            title,img = all_traces_list[slice_idx][seq_idx] # recall that [[sequence for a single slice]]
            ax  = axes[seq_idx][slice_idx]
            ax.imshow(img, cmap='gray', aspect='auto')
            ax.set_title(f"{title}\nSlice {slice_idx}")
            ax.axis('off')
    if suptitle:
        fig.suptitle(suptitle)
    plt.tight_layout()
    plt.show()

# Example usage:
# plot_random_slices_with_layers("results.npy", n_examples=3, layer_names=["BM", "RPE", "EZ"])

DEBUG_PLOTS = True          # flip to False to silence everything

def p(arr, title=None, cmap='gray', **imshow_kw):
    """
    Pass-through helper that *also* imshows if DEBUG_PLOTS is True.

    Example
    -------
    grad = p(np.abs(sobel(img, axis=0)), 'vertical grad')
    """
    if DEBUG_PLOTS:
        plt.figure(figsize=(4, 3))
        plt.imshow(arr, cmap=cmap, **imshow_kw,aspect='auto')
        if title:
            plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    return arr            # so you can assign it as usual


class PlotTracer:
    def __init__(self, show=True):
        self.show     = show        # whether to actually plt.show()
        self.tracing  = False       # turned on when you want to record
        self.images   = []          # list of (title, array)

    def __call__(self, arr, title=None, cmap='gray', **kw):
        # show inline as before
        if self.show:
            fig, ax = plt.subplots(figsize=(3,3))
            ax.imshow(arr, cmap=cmap, **kw,aspect='auto')
            if title: ax.set_title(title)
            ax.axis('off')
            plt.tight_layout()
            plt.show()

        # record if we're in tracing mode
        if self.tracing:
            # make a copy so later mutating `arr` doesn’t clobber
            self.images.append((title or "", arr.copy()))

        return arr
   
    ### NAPARI Utils
import numpy as np
from scipy.ndimage import map_coordinates

def flatten_image(img: np.ndarray, rpe_curve: np.ndarray, output_height: int = 256):
    """
    Shift each column so that the RPE curve becomes a flat line.
    
    Parameters
    ----------
    img : (H, W) image
    rpe_curve : (W,) array of row indices (float or int) representing the RPE
    output_height : int, how many rows to keep above the RPE line
    
    Returns
    -------
    flat_img : (output_height, W) image aligned to RPE
    """
    H, W = img.shape
    x_coords = np.arange(W)
    
    # Build sampling coordinates
    rows = np.arange(output_height)[:, None]  # shape (H_out, 1)
    cols = np.arange(W)[None, :]             # shape (1, W)
    
    # For each column, subtract offset from RPE row upward
    y_coords = rpe_curve[cols] - rows        # (H_out, W)

    # Clip to bounds
    y_coords = np.clip(y_coords, 0, H - 1)

    # Sample using map_coordinates (per-pixel bilinear)
    flat_img = map_coordinates(img, [y_coords, cols.repeat(output_height, axis=0)],
                               order=1, mode='reflect')
    
    return flat_img

def flatten_image_centered(idx:int ,img: np.ndarray, rpe_curve: np.ndarray, output_height: int = 400):
    """
    Flatten the image so that the RPE curve is centered vertically.

    Parameters
    ----------
    idx: can be dummy, just gets returned
    img : (H, W) input image
    rpe_curve : (W,) RPE row indices
    output_height : int, desired height of output image

    Returns
    -------
    flat_img : (output_height, W) image with RPE centered
    """
    H, W = img.shape
    center = output_height // 2

    # Build sampling coordinates: range [-center, ..., 0, ..., +offset]
    rows = np.arange(output_height)[:, None]  # shape (H_out, 1)
    offset_from_rpe = rows - center           # shift center of output to RPE

    # Repeat column indices (for X-coords)
    cols = np.arange(W)[None, :]              # shape (1, W)

    # Compute Y-coords in original image space
    y_coords = rpe_curve[cols] + offset_from_rpe  # shape (H_out, W)
    y_coords = np.clip(y_coords, 0, H - 1)

    # Sample
    x_coords = cols.repeat(output_height, axis=0)
    flat_img = map_coordinates(img, [y_coords, x_coords], order=1, mode='reflect')

    return idx,flat_img

import numpy as np

def flatten_other_curves(
    layers: dict[str, np.ndarray],
    ref_layer: str,
    output_height: int = 256,
    *,
    centre: int | None = None
) -> dict[str, np.ndarray]:
    """
    Return a new dict of flattened curves so that `ref_layer` is
    constant at `centre` (default = output_height//2), and all
    other layers are shifted by the same per-pixel offsets.

    Parameters
    ----------
    layers : dict of {layer_name: (n_slices, W) array}
    ref_layer : which key in `layers` to use as reference
    output_height : height of the hypothetical flattened image
    centre : row index where `ref_layer` should land.
             Defaults to output_height//2.

    Returns
    -------
    new_layers : dict of {layer_name: (n_slices, W) array}
    """
    # 1) Figure out the centre row
    if centre is None:
        centre = output_height // 2

    # 2) Pull out the reference curves
    if ref_layer not in layers:
        raise KeyError(f"{ref_layer!r} not in layers keys: {list(layers)}")
    ref = layers[ref_layer]  # shape = (n_slices, W)

    # 3) Compute per-slice, per-column offset:
    #    offset[z,x] = centre - ref[z,x]
    offset = centre - ref

    # 4) Build the flattened dict
    new_layers: dict[str, np.ndarray] = {}
    for name, arr in layers.items():
        if arr.shape != ref.shape:
            raise ValueError(f"Layer {name!r} shape {arr.shape} != ref shape {ref.shape}")
        if name == ref_layer:
            # reference becomes a constant line at `centre`
            new_layers[name] = np.full_like(arr, fill_value=centre, dtype=float)
        else:
            # shift every other curve by the same offset
            new_layers[name] = arr.astype(float) + offset

    return new_layers


from concurrent.futures import ProcessPoolExecutor
def flatten_volume(vol,layer,output_height=256,debug=False):
    if debug==True:
        output = []
        for idx in range(vol.shape[0]):
            idx,flat_img = flatten_image_centered(idx, vol[idx, :, :], layer[idx,:],output_height)
            output.append(flat_img)
        return np.stack(output,axis=0)


    with ProcessPoolExecutor(max_workers=12) as exe:
        futures = [
            exe.submit(flatten_image_centered, idx, vol[idx, :, :], layer[idx,:],output_height)
            for idx in range(vol.shape[0])
        ]
        # collect results and sort by bscan_idx
        results = [fut.result() for fut in futures]
    output = np.stack([flattened for idx,flattened in sorted(results, key=lambda x: x[0])],axis=0)
    return output



def build_layer_paths(name: str, layer: np.ndarray):
    """
    Build the list of (W×3) coordinates for every Z-slice in this layer.
    Returns (name, paths_list).
    """
    Z, W = layer.shape
    x = np.arange(W, dtype=float)
    paths = []
    for z in range(Z):
        y = layer[z].astype(float)
        # stack into (W,3): [Z, Y, X]
        coords = np.column_stack([
            np.full(W, z, dtype=float),
            y,
            x,
        ])
        paths.append(coords)
    return name, paths
