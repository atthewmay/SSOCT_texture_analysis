from __future__ import annotations
from typing import TYPE_CHECKING

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # adds Han_AIR/ to path
from matplotlib.backends.backend_pdf import PdfPages

if TYPE_CHECKING:
    from code_files.segmentation_code import segmentation_step_functions as ssf
    # or: import ... as ssf  (either is fine)

import matplotlib.pyplot as plt
import code_files.segmentation_code.segmentation_utility_functions as suf
import numpy as np
from PyPDF2 import PdfMerger
from matplotlib.backends.backend_pdf import PdfPages
import time
import textwrap


def to_rgb(img):
    """2d image input"""
    if img.ndim == 3:
        return img
    # normalize to [0,1] floats
    g = img.astype(np.float32)
    g -= g.min()
    if g.max()>0:
        g /= g.max()
    # stack into RGB
    return np.stack([g, g, g], axis=-1)
    # return np.repeat(img[..., None], 3, axis=2)   # (H,W)->(H,W,3)


def overlay_line_on_image(img, line, color=np.array([1,0,0]),thickness=1):
    """
    → Returns an (H, W, 3) float array in [0,1] with a red line overlaid.
    """
    H = img.shape[0]
    rgb = to_rgb(img)
    # draw red line
    for c, r in enumerate(line):
        if np.isnan(r): continue
        r=int(round(r))
        for t in range(-thickness//2, thickness//2 + 1):
            ri = r + t
            if 0 <= ri < H:
                rgb[ri,c,:] = color 
    return rgb

def overlay_peaks_on_image(img,peaks,color = np.array([1,0,0])):
    """peaks is like a list of np.arrays of column coordinates"""
    rgb = to_rgb(img)
    # draw pixels
    for c,peaks in enumerate(peaks):
        for row_idx in peaks:
            rgb[row_idx,c,:] = color 
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
    'rpe_smooth'           : dict(fmt='y-',  alpha = 0.6,lw=0.5, label='rigid'),
    'lower_edge_line'           : dict(fmt='r-',  alpha = 0.6,lw=0.5, label='lower_line (not smoothed)'),
    'rpe_guided'           : dict(fmt='c-',  lw=0.5, label='rigid'),
    'rpe_guided_tube_smoothed'           : dict(fmt='c-',  lw=0.5, label='rigid'),
    'ilm_seg'              : dict(fmt='m--', lw=0.5, label='ilm_seg'),
    'ilm_seg_flat'              : dict(fmt='m--', lw=0.5, label='ilm_seg_flat'),
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



# def render_PDF_page(pdf: PdfPages, title: str, dbg_ilm, dbg_rpe, original_line = None):
#     """Render one page with staged panels for RPE (row 1) and ILM (row 2)."""
#     # Define panel specs for RPE and ILM
#     rpe_panels = [
#         PanelSpec('original_image'),
#         PanelSpec('hypersmoothed_img',['ilm_seg_flat']),
#         PanelSpec('downsampled_img'),
#         PanelSpec('enh_f'),
#         PanelSpec('enh'),
#         PanelSpec('peak_suppressed'),
#         # PanelSpec('seeds'),
#         PanelSpec('prob'),
#         PanelSpec('edge'),
#         PanelSpec('guided_cost_raw'),
#         # PanelSpec('guided_cost'),
#         PanelSpec('guided_cost_raw_tube_smoothed'),
#         # PanelSpec('guided_cost_refined'),
#         PanelSpec('highres_diff_horiz_blur'),
#         PanelSpec('lower_edge_of_tubed'),
#         # PanelSpec('original_image', ['rpe_raw', 'rpe_guided_tube_smoothed','rpe_smooth']),
#         PanelSpec('original_image', ['rpe_raw', 'rpe_smooth','lower_edge_line']),
#     ]
#     ilm_panels = [
#         PanelSpec('img'),
#         PanelSpec('enh'),
#         PanelSpec('edge'),
#         PanelSpec('ilm_tube_cost_raw'),
#         PanelSpec('original_image', ['ilm_raw', 'ilm_smooth']),
#     ]

#     ncols = max(len(rpe_panels), len(ilm_panels))

#     fig, axes = plt.subplots(2, ncols, figsize=(2.6 * ncols, 6.5))

#     # RPE row
#     for j, spec in enumerate(rpe_panels):
#         ax = axes[0, j]
#         draw_panel(ax, dbg_rpe, spec)
#     if original_line is not None: # put the original RPE line on
#         print('plotting the original line too')
#         x = np.arange(len(original_line))
#         axes[0,len(rpe_panels)-1].plot(x,original_line, label='prior_plot')
#     #     ax.legend(fontsize=6, loc='upper right')
#     # ax.axis('off')
#     # ILM row
#     for j, spec in enumerate(ilm_panels):
#         ax = axes[1, j]
#         draw_panel(ax, dbg_ilm, spec)

#     # Any spare axes (if panel lists differ)
#     for j in range(len(rpe_panels), ncols):
#         axes[0, j].axis('off')
#     for j in range(len(ilm_panels), ncols):
#         axes[1, j].axis('off')

#     fig.suptitle(title, fontsize=10)
#     fig.tight_layout(rect=(0, 0, 1, 0.97))
#     pdf.savefig(fig)
#     plt.close(fig)

    
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

def _draw_section(fig, outer_spec, panels, dbg,  max_cols, title=None):
    n = len(panels)
    ncols = max_cols
    nrows = max(1, math.ceil(n / ncols))

    inner = GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=outer_spec, wspace=0.15, hspace=0.15)

    axes = []
    for k in range(nrows * ncols):
        ax = fig.add_subplot(inner[k // ncols, k % ncols])
        axes.append(ax)
        if k < n:
            draw_panel(ax, dbg, panels[k])
        else:
            ax.axis("off")

    if title:
        # simple section label (top-left)
        axes[0].set_title(title, fontsize=9, loc="left")

    return axes, (nrows, ncols)

def render_two_sections(
    rpe_panels, ilm_panels,
    dbg_rpe, dbg_ilm,
    pdf,
    title="",
    max_cols=6,
    figsize_per_col=4,
    row_height=4.1,
    original_line=None,
    original_line_on="rpe_first",  # "rpe_first" or "rpe_last"
):
    # compute heights from each section's needed rows
    rpe_rows = max(1, math.ceil(len(rpe_panels) / max_cols))
    ilm_rows = max(1, math.ceil(len(ilm_panels) / max_cols))
    fig_h = row_height * (rpe_rows + ilm_rows) + 0.8
    fig_w = figsize_per_col * max_cols

    fig = plt.figure(figsize=(fig_w, fig_h))
    outer = GridSpec(2, 1, height_ratios=[rpe_rows, ilm_rows], hspace=0.25, figure=fig)

    rpe_axes, _ = _draw_section(fig, outer[0], rpe_panels, dbg_rpe,  max_cols, title="RPE")
    ilm_axes, _ = _draw_section(fig, outer[1], ilm_panels, dbg_ilm,  max_cols, title="ILM")

    # # optional overlay: original line on one of the RPE panels
    # if original_line is not None and len(rpe_panels) > 0:
    #     ax = rpe_axes[0] if original_line_on == "rpe_first" else rpe_axes[min(len(rpe_panels)-1, len(rpe_axes)-1)]
    #     x = np.arange(len(original_line))
    #     ax.plot(x, original_line, label="prior_plot")
    #     # ax.legend(fontsize=6, loc="upper right")

    if title:
        fig.suptitle(title, fontsize=10)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    pdf.savefig(fig)
    plt.close(fig)

def render_PDF_page(pdf: PdfPages, title: str, dbg_ilm, dbg_rpe, original_line = None):
    rpe_panels = [
        PanelSpec('original_image'),
        PanelSpec('hypersmoothed_img',['ilm_seg_flat']),
        PanelSpec('downsampled_img'),
        PanelSpec('enh_f'),
        PanelSpec('enh'),
        PanelSpec('peak_suppressed'),
        # PanelSpec('seeds'),
        PanelSpec('prob'),
        PanelSpec('edge'),
        PanelSpec('guided_cost_raw'),
        # PanelSpec('guided_cost'),
        PanelSpec('guided_cost_raw_tube_smoothed'),
        # PanelSpec('guided_cost_refined'),
        PanelSpec('highres_diff_horiz_blur'),
        PanelSpec('lower_edge_of_tubed'),
        # PanelSpec('original_image', ['rpe_raw', 'rpe_guided_tube_smoothed','rpe_smooth']),
        PanelSpec('original_image', ['rpe_raw', 'rpe_smooth','lower_edge_line']),
    ]
    ilm_panels = [
        PanelSpec('img'),
        PanelSpec('enh'),
        PanelSpec('edge'),
        PanelSpec('ilm_tube_cost_raw'),
        PanelSpec('original_image', ['ilm_raw', 'ilm_smooth']),
    ]
    render_two_sections(rpe_panels,ilm_panels,dbg_rpe,dbg_ilm,pdf=pdf,title=title)
    fig,ax = plt.subplots()
    draw_panel(ax,dbg_rpe,rpe_panels[-1])
    plt.show()

def render_PDF_page_ArrayBoard(pdf:PdfPages,title: str,ctx_ilm:ssf.ILMContext,ctx_rpe:ssf.RPEContext):
    """using the updated and simplified arrayboard interface instead. """


    AB = ArrayBoard(plt_display=True,return_fig=True)
    # Add the RPE plots
    if 'gradient_line' in ctx_rpe.hypersmoother_params.hypersmoother_path_extras:
        AB.add(ctx_rpe.original_image,lines={'hypersmoothed':ctx_rpe.hypersmoother_params.hypersmoother_path,'line via upgrad':ctx_rpe.hypersmoother_params.hypersmoother_path_extras['gradient_line']},title="original")
    else:
        AB.add(ctx_rpe.original_image,lines={'hypersmoothed':ctx_rpe.hypersmoother_params.hypersmoother_path},title="original")
    AB.add(ctx_rpe.hypersmoother_params.coarse_hypersmoothed_img,title="hypersmooth_coarse")
    AB.add(ctx_rpe.hypersmoothed_img,title="hypersmoothed_img")
    AB.add(ctx_rpe.downsampled_img,title="downsampled_img")
    AB.add(ctx_rpe.peak_suppressed,title="original_peak_suppressed")
    AB.add(ctx_rpe.peak_suppressed,title="peak_suppressed")
    AB.add(ctx_rpe.prob,title="prob")
    AB.add(ctx_rpe.edge,title="edge")
    AB.add(ctx_rpe.guided_cost_raw,title="guided_cost_raw")
    AB.add(ctx_rpe.guided_cost_raw_tube_smoothed,title="guided_cost_raw_tube_smoothed")
    AB.add(ctx_rpe.original_image,lines = {"hypersmoothed":ctx_rpe.hypersmoother_params.hypersmoother_path, "rpe_smooth":ctx_rpe.rpe_smooth},title="original with rpe_smooth")
    # now the highres stuff
    AB.add(ctx_rpe.highres_ctx.diff_down_up,title="highfreq_diff_down_up")
    AB.add(ctx_rpe.highres_ctx.lower_edge_of_tubed,title="lower_edge_of_tubed")
    if hasattr(ctx_rpe,'highres_suppressed'): # A logged history version
        AB.add(ctx_rpe.highres_suppressed,title="tubed_and_suppressed")
    AB.add(ctx_rpe.highres_ctx.lower_edge_of_tubed,lines = {'rpe_flat':ctx_rpe.flat_rpe_smooth,"rpe_raw":ctx_rpe.flat_highres_rpe_raw,"rpe_refined":ctx_rpe.flat_highres_rpe_refined},title="lower_edge_of_tubed with rpe raw (lower edge) and refined with DP")
    
    AB.add(ctx_rpe.highres_ctx.lower_edge_of_tubed,lines = {"rpe_raw":ctx_rpe.flat_highres_rpe_raw,"rpe_refined":ctx_rpe.flat_highres_rpe_refined},title="lower_edge_of_tubed with rpe raw (lower edge) and refined with DP")
    AB.add(ctx_rpe.highres_ctx.highfreq_diff_down_up,title="highest-res gradient")
    AB.add(ctx_rpe.highres_ctx.guide_image,title="DP_guide_img")

    AB.add(ctx_rpe.original_image,lines = {"hypersmoothed":ctx_rpe.hypersmoother_params.hypersmoother_path,
                                            "rpe_smooth":ctx_rpe.rpe_smooth,
                                            "rpe_refined1":ctx_rpe.highres_ctx.rpe_refined,
                                            "rpe_refined2":ctx_rpe.highres_ctx.rpe_refined2,
                                            "rpe_smooth2":ctx_rpe.highres_ctx.rpe_smooth2,
                                            },title="Original with all lines")


    # ctx.highres_ctx.guide_image  = guide_image 
    # ctx.highres_ctx.highfreq_diff_down_up = highfreq_diff_down_up
    # ctx.highres_ctx.rpe_refined2 = rpe_refined2

    # AB.add(ctx_rpe.highres_ctx.highfreq_diff_down_up,title="highfreq_diff_down_up")
    fig = AB.render(suptitle=title)
    pdf.savefig(fig)
    # AB.add(ctx_rpe.hypersmoothed_img,title="hypersmoothed_img")

def save_results_in_PDF(results,output_file=None):
    if output_file is None:
        DEFAULT_REPORT_DIR = (Path(__file__).resolve().parents[2] / "reports")  # repo_root/reports
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = DEFAULT_REPORT_DIR / 'seg_reports' / f"seg_report_{timestamp}.pdf"
    print(f"will save report pdf at {output_file}")

    with PdfPages(output_file) as pdf:
        for i,(order_idx, dbg_ilm, dbg_rpe,work_id) in enumerate(results):
            # title = 'Current slice' if order_idx == 0 else "None title"
            title = work_id
            # render_PDF_page(pdf, title, dbg_ilm, dbg_rpe)
            # render_PDF_page(pdf, title, dbg_ilm, dbg_rpe)
            render_PDF_page_ArrayBoard(pdf, title, dbg_ilm, dbg_rpe)
            # render_two_sections(pdf, title, dbg_ilm, dbg_rpe)

    # Clean any sweep PDFs produced elsewhere
    try:
        clean_sweep_pdfs()
    except Exception:
        pass



import math

class ArrayBoard:
    def __init__(self,skip=False,plt_display=False,dpi = 300,save_tag="",return_fig=False,ncols_max=6):
        self.items = []  # list of (array, title)
        self.skip = skip
        self.plt_display = plt_display
        self.dpi = dpi
        self.save_tag = save_tag
        self.return_fig = return_fig
        self.ncols_max = ncols_max
            

    def add(self, arr,lines=dict(), title=None):
        """lines can be an empty dict or dict with 'title' line"""
        if self.skip:
            return
        self.items.append(["image",np.asarray(arr),lines, title])
        return self  # enables chaining

    def add_plot(self, draw_fn, title=None):
        """draw_fn(ax): you draw anything you want on ax..
        use like 
        AB.add_plot(
            lambda ax: MyClass.plot_ascan(ax, signal_1d, peak_rows, title="A-scan"),
            title="ascan"
        )
        """
        if self.skip: return
        self.items.append(("plot", draw_fn, {}, title))
        return self

    def clear(self):
        self.items.clear()

    def render(self, ncols=None, figsize_per_cell=(4, 4), suptitle=None):
        if self.skip:
            return
        n = len(self.items)
        if n == 0:
            return None, None

        if ncols is None:
            ncols = min(self.ncols_max, n)
        nrows = math.ceil(n / ncols)

        fig, axs = plt.subplots(
            nrows, ncols,
            figsize=(figsize_per_cell[0] * ncols, figsize_per_cell[1] * nrows),
            squeeze=False,
            dpi = self.dpi,
        )

        for i, (kind,payload,lines, title) in enumerate(self.items):
            if payload is None or (isinstance(payload, np.ndarray) and payload.shape == () and payload.dtype == object and payload.item() is None):
                print(f"payload for {title} is effectively none, coninuing")
                continue
            r, c = divmod(i, ncols)
            ax = axs[r][c]
            if kind == "image":
                ax.imshow(payload, cmap="gray", aspect="auto")
                for name,line in lines.items():
                    st = LAYER_STYLE.get(name)
                    if st:
                        ax.plot(line, st['fmt'],
                                lw=st['lw'], label=st['label'])
                        ax.legend(fontsize=6, loc='upper right')
                    else:
                        try:
                            ax.plot(line,alpha=0.7,lw=0.5,label=name)
                        except:
                            print(f"failed at {name} on title={title}")
                            raise Exception
                        ax.legend(fontsize=6, loc='upper right')
            elif kind == "plot":
                payload(ax)
            else:
                raise Exception

            if title:
                ax.set_title(title, fontsize=10)

        # hide any unused axes
        for j in range(n, nrows * ncols):
            r, c = divmod(j, ncols)
            axs[r][c].axis("off")

        if suptitle:
            fig.suptitle(suptitle)

        fig.tight_layout()
        print(f"self.plt_display = {self.plt_display}")
        if self.return_fig:
            return fig
        if self.plt_display:
            plt.show()
        else:
            self.save_tag
            temp_fig_location = f"/Users/matthewhunt/Research/Iowa_Research/Han_AIR/results/temp_figs/ArrayBoardDisplay{self.save_tag}"
            plt.savefig(temp_fig_location)
            print(f"will be saving pdf figure at {temp_fig_location}")
        # return fig, axs

from itertools import product

def dict_product(grid: dict):
    """Yield dicts for the cartesian product of a param grid."""
    keys = list(grid)
    for vals in product(*(grid[k] for k in keys)):
        yield dict(zip(keys, vals))

# def sweep_to_arrayboard(board, fn, *, base_kwargs: dict, grid: dict,
#                         title_fn=None, metric_fn=None):
#     """
#     Runs fn(**base_kwargs, **params) for each params in grid and adds output to board.
#     Returns records (params + metric) for easy plotting.

#     - title_fn(params) -> str   (optional)
#     - metric_fn(out, params) -> float|dict  (optional)
#     """
#     records = []
#     for params in dict_product(grid):
#         kwargs = {**base_kwargs, **params}
#         out = fn(**kwargs)

#         title = title_fn(params) if title_fn else ", ".join(f"{k}={params[k]}" for k in grid)
#         title = textwrap.fill(title,40)
#         board.add(out, title=title)

#         if metric_fn is not None:
#             m = metric_fn(out, params)
#             records.append({**params, **(m if isinstance(m, dict) else {"metric": m})})
#         else:
#             records.append({**params})
#     return records

    

def quickfig(array,title=None):
    plt.figure()
    plt.imshow(array,cmap='gray')
    plt.title(title)
    try:
        plt.show()
    except:
        save_path = "/Users/matthewhunt/Research/Iowa_Research/Han_AIR/results/temp_figs/save_quickfig"
        print(f'unable to plot apparanetly, saving instead at {save_path}')
        plt.savefig(save_path)


from concurrent.futures import ProcessPoolExecutor
import textwrap

def _sweep_worker(job):
    fn, base_kwargs, params = job
    kwargs = {**base_kwargs, **params}
    out = fn(**kwargs)
    return params, out

def sweep_to_arrayboard(board, fn, *, base_kwargs: dict, grid: dict,
                        title_fn=None, metric_fn=None, nworkers=1):
    """Note this will fail if using a lambda and nworkers>1"""
    records = []
    params_list = list(dict_product(grid))

    if nworkers <= 1:
        for params in params_list:
            out = fn(**{**base_kwargs, **params})
            title = title_fn(params) if title_fn else ", ".join(f"{k}={params[k]}" for k in grid)
            board.add(out, title=textwrap.fill(title, 40))
            if metric_fn is not None:
                m = metric_fn(out, params)
                records.append({**params, **(m if isinstance(m, dict) else {"metric": m})})
            else:
                records.append({**params})
        return records

    jobs = [(fn, base_kwargs, params) for params in params_list]
    with ProcessPoolExecutor(max_workers=nworkers) as ex:
        for params, out in ex.map(_sweep_worker, jobs, chunksize=1):
            title = title_fn(params) if title_fn else ", ".join(f"{k}={params[k]}" for k in grid)
            board.add(out, title=textwrap.fill(title, 40))
            if metric_fn is not None:
                m = metric_fn(out, params)
                records.append({**params, **(m if isinstance(m, dict) else {"metric": m})})
            else:
                records.append({**params})
    return records
