import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # adds Han_AIR/ to path
import matplotlib.pyplot as plt
import code_files.segmentation_code.segmentation_utility_functions as suf
import numpy as np
from PyPDF2 import PdfMerger

def overlay_line_on_image(gray_img, line, thickness=1):
    """
    → Returns an (H, W, 3) float array in [0,1] with a red line overlaid.
    """
    H, W = gray_img.shape
    # normalize to [0,1] floats
    g = gray_img.astype(np.float32)
    g -= g.min()
    if g.max()>0:
        g /= g.max()
    # stack into RGB
    rgb = np.stack([g, g, g], axis=-1)

    # draw red line
    for c, r in enumerate(line.astype(int)):
        if np.isnan(r): continue
        for t in range(-thickness//2, thickness//2 + 1):
            ri = r + t
            if 0 <= ri < H:
                rgb[ri, c, 0] = 1.0   # red channel
                rgb[ri, c, 1] = 0.0
                rgb[ri, c, 2] = 0.0

    return rgb


def overlay_helper(line,img_to_overlay,upsample_d_vert = None):
    """takes raw input (img or line) and calcs a line, then overlays on the image"""
    if upsample_d_vert:
        line = suf.upsample_path(line,vertical_factor=upsample_d_vert,original_length=512)
    rgb_img = overlay_line_on_image(img_to_overlay,line)
    return rgb_img


def save_image_exploration(downsampled_img,rpe_guided_tube_smoothed,pickle_save = False):
    """a quickplot function"""

    if downsampled_img is None and rpe_guided_tube_smoothed is None:
        import pickle
        pickle_path = "/Users/matthewhunt/Research/Iowa_Research/Han_AIR/results/temp_pickle/img_exploration_stuff"
        [downsampled_img,rpe_guided_tube_smoothed] = pickle.load(open(pickle_path,'rb'))
    elif pickle_save:
        print('gonna pickle')
        import pickle
        import os
        pickle_path = "/Users/matthewhunt/Research/Iowa_Research/Han_AIR/results/temp_pickle/img_exploration_stuff"
        pickle.dump([downsampled_img,rpe_guided_tube_smoothed],open(pickle_path,'wb'))

    img_dict = {'downsampled_img':downsampled_img}


    img_dict['enhT'] = suf._boundary_enhance(downsampled_img,vertical_kernel_size=5,dark2bright=True)

    img_dict['enhF'] = suf._boundary_enhance(downsampled_img,vertical_kernel_size=5,dark2bright=False)

    import math
    ncols = 3
    nrows = math.ceil((len(img_dict)+2)/ncols)
    fig,ax = plt.subplots(nrows,ncols,figsize = (ncols*3,3*nrows))
    ax = ax.flatten()

    line_dict = {}

    plot_idx = 0
    ax[plot_idx].imshow(downsampled_img,cmap='gray',aspect='auto')
    plot_idx += 1
    for i,(k,v) in enumerate(img_dict.items()):
        ax[plot_idx].imshow(v,cmap='gray',aspect='auto')
        ax[plot_idx].set_title(k)
        ax[plot_idx].plot(rpe_guided_tube_smoothed,c='c',alpha=0.4,lw=0.5,label='original')
        if i != 0:
            custom_line,_,cost_img = suf.tube_smoother_DP(img=v,
                                            guide_y=rpe_guided_tube_smoothed,
                                            lambda_step=0.1,
                                            sigma_guide=20)
            line_dict[k] = custom_line
            ax[plot_idx].plot(custom_line,alpha=0.4,lw=0.5,c='r')
            plot_idx += 1
            ax[plot_idx].imshow(cost_img,cmap='gray',aspect='auto')
            ax[plot_idx].plot(custom_line,alpha=0.4,lw=0.5,c='r')
        ax[plot_idx].axis('off')
        plot_idx += 1


    for k,v in line_dict.items():
        ax[1].plot(v,label=k,alpha=0.4,lw=0.5)
    ax[1].legend()
    fig.tight_layout()
    plt.savefig("/Users/matthewhunt/Research/Iowa_Research/Han_AIR/results/temp_figs/save_a_fig")



from typing import NamedTuple, Sequence, Optional
class PanelSpec(NamedTuple):
    layer: str                     # which background to show
    overlays: Optional[Sequence[str]] = None   # curves to plot (or None)


PDF_ROOT = "/Users/matthewhunt/Research/Iowa_Research/Han_AIR/results/temp_pdfs"
def clean_sweep_pdfs(base_dir=PDF_ROOT):
    """
    For each subdirectory under base_dir, merge all .pdfs into
    one named <subdir>.pdf in base_dir, then delete that subdir.
    """
    base = Path(base_dir)
    if not base.is_dir():
        raise FileNotFoundError(f"{base} does not exist")

    for sub in sorted(base.iterdir()):
        if not sub.is_dir():
            continue
        pdfs = sorted(sub.glob("*.pdf"))
        if not pdfs:
            continue

        merger = PdfMerger()
        for p in pdfs:
            merger.append(str(p))
        out_path = base / f"{sub.name}.pdf"
        merger.write(str(out_path))
        merger.close()

        # clean up individual pages & directory
        for p in pdfs:
            # print(f"will delete {p}")
            p.unlink()
        sub.rmdir()


LAYER_STYLE = {
    'rpe_raw'              : dict(fmt='m--', lw=0.5, label='raw'),
    'rpe_smooth'           : dict(fmt='y-',  lw=0.5, label='rigid'),
    'rpe_guided'           : dict(fmt='c-',  lw=0.5, label='rigid'),
    'rpe_guided_tube_smoothed'           : dict(fmt='c-',  lw=0.5, label='rigid'),
    'ilm_raw'              : dict(fmt='m--', lw=0.5, label='raw'),
    'ilm_smooth'           : dict(fmt='y-',  lw=0.5, label='rigid'),
    'DP_enh_vertical_window': dict(fmt='c--', lw=1, label='DP enh-win'),
    'DP_enh_sobel'          : dict(fmt='r--', lw=1, label='DP enh-sobel'),
    'DP_img_vertical_window': dict(fmt='c--', lw=1, label='DP img-win'),
    'DP_img_sobel'          : dict(fmt='r--', lw=1, label='DP img-sobel'),
}
# ---------------------------------------------------------------

def draw_panel(ax, dbg, spec: PanelSpec):
    """Draw one subplot.
    """
    # accept either 'enh' or Panel('enh', [...])
    if isinstance(spec, str):
        spec = PanelSpec(spec, None)

    # 1) background image
    cmap = 'hot' if spec.layer == 'prob' else 'gray'
    ax.imshow(getattr(dbg, spec.layer), cmap=cmap, aspect='auto')
    ax.set_title(spec.layer)

    # 2) overlay curves (only those requested *and* present in dbg)
    if spec.overlays:
        x = np.arange(getattr(dbg, spec.layer).shape[1])
        for name in spec.overlays:
            if hasattr(dbg, name) and getattr(dbg, name) is not None: # Bc we allow optional attrs, must check if such an overlay exists
                st = LAYER_STYLE[name]
                ax.plot(x, getattr(dbg, name), st['fmt'],
                        lw=st['lw'], label=st['label'])
        ax.legend(fontsize=6, loc='upper right')
    ax.axis('off')

