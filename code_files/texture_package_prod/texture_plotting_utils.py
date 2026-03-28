#reviewed
from __future__ import annotations

from pathlib import Path
import pdb

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


from code_files.segmentation_code.segmentation_plot_utils import ArrayBoard

from .texture_extraction_utilities import DenseMapMeta, resample_map_to_image


def _show(ax, image: np.ndarray, title: str, cmap: str = 'gray', box_aspect: float | None = 1.0):
    ax.imshow(image, cmap=cmap, aspect='auto' if box_aspect is None else 'equal')
    if box_aspect is not None:
        ax.set_box_aspect(box_aspect)
    ax.set_title(title, fontsize=9)
    ax.axis('off')


def plot_feature_mosaic(
    base_image: np.ndarray,
    feature_maps: dict[str, np.ndarray],
    meta: DenseMapMeta | None = None,
    max_features: int = 20,
    out_path: str | Path | None = None,
):
    names = list(feature_maps)[:max_features]
    arrays = [feature_maps[n] if meta is None else resample_map_to_image(feature_maps[n], meta) for n in names]

    if ArrayBoard is not None and meta is None and base_image.ndim == 2:
        board = ArrayBoard(plt_display=False, return_fig=True, ncols_max=4,save_tag=f"AB_texture_test")
        board.add(base_image, title='base_image')
        for name, arr in zip(names, arrays):
            board.add(arr, title=name)
        fig = board.render(suptitle='texture feature mosaic')
    else:
        n = len(arrays) + 1
        ncols = min(4, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), dpi=180)
        axes = np.atleast_1d(axes).ravel()
        _show(axes[0], base_image, 'base_image')
        for ax, name, arr in zip(axes[1:], names, arrays):
            _show(ax, arr, name)
        for ax in axes[n:]:
            ax.axis('off')
        fig.tight_layout()

    if out_path is not None:
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
    return fig


def plot_regions_overlay(image: np.ndarray, masks: dict[str, np.ndarray], out_path: str | Path | None = None):
    fig, ax = plt.subplots(figsize=(7, 7), dpi=180)
    ax.imshow(image, cmap='gray')
    colors = [
        (1.0, 1.0, 1.0), (1.0, 0.5, 0.0), (0.0, 1.0, 1.0), (1.0, 0.0, 1.0),
        (0.5, 1.0, 0.0), (1.0, 0.2, 0.2), (0.2, 0.8, 1.0)
    ]
    ring_keys = [k for k in masks if k == 'center' or k.endswith('_ring') or k.startswith('extra_ring_')]
    for i, key in enumerate(ring_keys):
        edge = masks[key] ^ ndimage.binary_erosion(masks[key])
        rgba = np.zeros((*image.shape[:2], 4), dtype=np.float32)
        rgba[edge, :3] = colors[i % len(colors)]
        rgba[edge, 3] = 0.95
        ax.imshow(rgba)
    ax.set_title('ETDRS + wider rings')
    ax.set_box_aspect(1)
    ax.axis('off')
    if out_path is not None:
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
    return fig


def plot_etdrs_overlay(image: np.ndarray, masks: dict[str, np.ndarray], out_path: str | Path | None = None):
    return plot_regions_overlay(image, masks, out_path=out_path)


def plot_alignment_preview(left: np.ndarray, right: np.ndarray, out_path: str | Path | None = None):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=180)
    _show(axes[0], left, 'left')
    _show(axes[1], right, 'right')
    overlay = np.zeros((*left.shape, 3), dtype=np.float32)
    overlay[..., 0] = (left - np.nanmin(left)) / max(np.nanmax(left) - np.nanmin(left), 1e-6)
    overlay[..., 1] = (right - np.nanmin(right)) / max(np.nanmax(right) - np.nanmin(right), 1e-6)
    axes[2].imshow(overlay)
    axes[2].set_title('overlay')
    axes[2].set_box_aspect(1)
    axes[2].axis('off')
    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)
    return fig




# def plot_texture_zarr_feature_grid(
#     zarr_root,
#     features=('raw__mean', 'raw__std', 'raw__glcm_contrast'),
#     n_slices=5,
#     save_tag="",
#     max_per_array_board = 60
# ):
#     import zarr

#     features = [f for f in features if f in zarr_root]

#     if not features:
#         raise ValueError('None of the requested features were found in the zarr.')
#     print(list(zarr_root.keys()))
#     print(f"all the features to be plotted are {features}")

#     Z = zarr_root[features[0]].shape[0]
#     z_indices = np.linspace(0, Z - 1, n_slices).round().astype(int)
#     z_indices = np.unique(z_indices)


#     for z in z_indices:
#         AB = ArrayBoard(plt_display=False, return_fig=True, ncols_max=len(features), save_tag=save_tag+"_slice_{i}")
#         for feat in features:
#             AB.add(zarr_root[feat][int(z), :, :], title=f'{feat} | z={int(z)}')

#         fig = AB.render()
#     return fig

def plot_texture_zarr_feature_grid(
        zarr_root,
        features=('raw__mean', 'raw__std', 'raw__glcm_contrast'),
        n_slices=5,
        save_tag="",
        max_per_array_board=60,
    ):
    import numpy as np

    features = [f for f in features if f in zarr_root]
    if not features:
        raise ValueError('None of the requested features were found in the zarr.')

    print(list(zarr_root.keys()))
    print(f"all the features to be plotted are {features}")

    Z = zarr_root[features[0]].shape[0]
    z_indices = np.linspace(0, Z - 1, n_slices).round().astype(int)
    z_indices = np.unique(z_indices)

    figs = []

    for z in z_indices:
        for start in range(0, len(features), max_per_array_board):
            feat_chunk = features[start:start + max_per_array_board]

            AB = ArrayBoard(
                plt_display=False,
                ncols_max=min(len(feat_chunk), 6),
                save_tag=f"{save_tag}_z{int(z):03d}_part{start // max_per_array_board:02d}",
            )

            for feat in feat_chunk:
                AB.add(
                    zarr_root[feat][int(z), :, :],
                    title=f'{feat} | z={int(z)}',
                )

            fig = AB.render()
            figs.append(fig)

    return figs