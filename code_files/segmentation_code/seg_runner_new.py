from __future__ import annotations
import argparse
import sys
from pathlib import Path
import hashlib

# Ensure repo root is on path (this file lives in repo_root/scripts)
sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import cv2
import file_utils as fu
# print(sys.path)


import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import code_files.segmentation_code.segmentation_plot_utils as spu
import code_files.segmentation_code.segmentation_pipelines as sp

import dask.array as da

# -----------------------------
#  Sampling helpers
# -----------------------------

def _stable_hash_u32(s: str, seed: int = 42) -> int:
    h = hashlib.blake2b((s + str(seed)).encode('utf-8'), digest_size=4)
    return int.from_bytes(h.digest(), 'little')


def pick_consistent_indices(n: int, k: int, key: str, seed: int = 42) -> np.ndarray:
    """Deterministically pick k unique indices from [0, n) based on a string key."""
    rs = np.random.RandomState(_stable_hash_u32(key, seed=seed))
    k = min(k, n)
    return rs.choice(n, size=k, replace=False)

def image_to_annotation_path(img_path: Path, ann_root: Path=None) -> Path:
    if ann_root is None:
        ann_root = Path("/Users/matthewhunt/Research/Iowa_Research/Han_AIR/testing_annotations")
    cand1 = ann_root / img_path.with_suffix(".labels.zarr").name
    cand2 = ann_root / img_path.with_suffix(".zarr").name
    return cand1 if cand1.exists() else cand2

def load_extra_slices(images_root: Path, total_k: int,golden_set_dict = None) -> list[np.ndarray]:
    """Used to use the mini_root, but don't need this w/ very fast loading from zarr.
    optiont to either:
    Load ~total_k slices by picking 1 per volume (deterministic per path) until we have total_k.
    or -- use some golden set I'll define in a yaml file with volume id"""
    if golden_set_dict is not None:
        print(f"will use entire golden set rather than the state {total_k}")
        total_k = 1e6
    images_root = Path(images_root)
    # vols = sorted(images_root.rglob('*.npy'))
    vols = sorted(images_root.rglob('*.img'))
    slices = []
    ONH_annotations = []
    stem_idx_order_combos = []
    for i,fp in enumerate(vols):
        vol = fu.load_ss_volume2(fp,mmap=True) # should be fast and memory light?
        try:
            annotation_root = Path("/Users/matthewhunt/Research/Iowa_Research/Han_AIR/testing_annotations")
            # annotation_path = annotation_root / fp.with_suffix('.zarr').name
            annotation_path = image_to_annotation_path(fp,annotation_root)
            # import pdb; pdb.set_trace()
            ONH_info = fu.load_ss_volume2(annotation_path,mmap=True) # should be fast and memory light?)
        except:
            print(f"unable to load ONH_info for {fp}")
            ONH_info = None
        if len(slices) >= total_k:
            break
        if golden_set_dict is None:
            z_idxs = [int(pick_consistent_indices(vol.shape[0], 1, key=str(fp))[0])] # A single element list for consistency
        else:
            z_idxs = golden_set_dict[fp.stem] # A dictionary 
            import pdb; pdb.set_trace()

        for z_idx in z_idxs:
            stem_idx_order_combos.append((fp.stem,z_idx))
            slices.append(vol[z_idx, :, :])
            if ONH_info is not None:
                ONH_annotations.append(ONH_info[z_idxs,:,:][...])
            else:
                ONH_annotations.append(None)
    return slices,ONH_annotations,stem_idx_order_combos


def load_golden_set_layers(golden_dict,stem_idx_order_combos,layers_root: Path=None):
    """This loads the bscans layers from the layers_root directory. 
    Ah, this fails bc it doesn't load in the same order as the above slices..."""
    layers_out = []
    if layers_root is None:
        layers_root = Path('/Users/matthewhunt/Research/Iowa_Research/Han_AIR/data_all_volumes_layers_08_23_25/')
    for stem,idxs in golden_dict.items():
        layer_volume = fu.load_layers(Path(layers_root)/f"{Path(stem).with_suffix('')}_layers.npz")
        for idx in idxs:
            print(idx)
            layers_out.append(layer_volume['rpe_smooth'][idx,:]) # take entire bscan layer
    return layers_out

def load_extra_slices_and_prior_layers(golden_set_dict,
                                       images_root: Path,
                                       layers_root: Path = Path('/Users/matthewhunt/Research/Iowa_Research/Han_AIR/data_all_volumes_layers_08_23_25/')):
    """unifying the above functions as a better refactor. """
    vols = sorted(images_root.rglob('*.img'))
    slices = []
    golden_set_layers = []
    ONH_annotations = []
    for i,fp in enumerate(vols):
        vol = fu.load_ss_volume2(fp,mmap=True) # should be fast and memory light?
        layer_volume = fu.load_layers(Path(layers_root)/f"{fp.stem}_layers.npz")

        annotation_root = Path("/Users/matthewhunt/Research/Iowa_Research/Han_AIR/testing_annotations")
        annotation_path = image_to_annotation_path(fp,annotation_root)
        ONH_info = da.from_zarr(annotation_path) # should be fast and memory light?)

        z_idxs = golden_set_dict[fp.stem] # A dictionary 
        for z_idx in z_idxs:
            slices.append(vol[z_idx, :, :])
            golden_set_layers.append(layer_volume['rpe_smooth'][z_idx,:]) # take entire bscan layer
            ONH_annotations.append(ONH_info[z_idx,:,:][...])
    return slices,ONH_annotations,golden_set_layers


def grab_current_onh_info(onh_path,z_index):
    """if it exists, also load up the ONH info.
    issue: these are stored as zarr directories, so we need to use a different function to load them"""
    print("you are loadign the full-resolution ONH info stack, may need downsampling if using a downsampled array")
    ONH_info = da.from_zarr(str(onh_path))                # lazy
    return ONH_info[int(z_index),:,:][...] # This is causing error somehow

# -----------------------------
#  Rendering
# -----------------------------

def render_page(pdf: PdfPages, title: str, dbg_ilm, dbg_rpe, original_line = None):
    """Render one page with staged panels for RPE (row 1) and ILM (row 2)."""
    # Define panel specs for RPE and ILM
    rpe_panels = [
        spu.PanelSpec('img'),
        spu.PanelSpec('enh_f'),
        spu.PanelSpec('enh'),
        spu.PanelSpec('peak_suppressed'),
        spu.PanelSpec('seeds'),
        spu.PanelSpec('prob'),
        spu.PanelSpec('edge'),
        spu.PanelSpec('guided_cost_raw'),
        spu.PanelSpec('guided_cost'),
        spu.PanelSpec('guided_cost_raw_tube_smoothed'),
        # spu.PanelSpec('guided_cost_refined'),
        spu.PanelSpec('original_image', ['rpe_raw', 'rpe_guided_tube_smoothed','rpe_smooth']),
    ]
    ilm_panels = [
        spu.PanelSpec('img'),
        spu.PanelSpec('enh'),
        spu.PanelSpec('edge'),
        spu.PanelSpec('ilm_tube_cost_raw'),
        spu.PanelSpec('original_image', ['ilm_raw', 'ilm_smooth']),
    ]

    ncols = max(len(rpe_panels), len(ilm_panels))

    fig, axes = plt.subplots(2, ncols, figsize=(2.6 * ncols, 6.5))

    # RPE row
    for j, spec in enumerate(rpe_panels):
        ax = axes[0, j]
        spu.draw_panel(ax, dbg_rpe, spec)
    if original_line is not None: # put the original RPE line on
        print('plotting the original line too')
        x = np.arange(len(original_line))
        axes[0,len(rpe_panels)-1].plot(x,original_line, label='prior_plot')
    #     ax.legend(fontsize=6, loc='upper right')
    # ax.axis('off')
    # ILM row
    for j, spec in enumerate(ilm_panels):
        ax = axes[1, j]
        spu.draw_panel(ax, dbg_ilm, spec)

    # Any spare axes (if panel lists differ)
    for j in range(len(rpe_panels), ncols):
        axes[0, j].axis('off')
    for j in range(len(ilm_panels), ncols):
        axes[1, j].axis('off')

    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    pdf.savefig(fig)
    plt.close(fig)

def golden_set_differences(results_extra_layers,golden_set_layers):
    """inputs are the results output which includs the dbg (debug outputs)
    and then we compare against the golden_set_layers"""
    scores = []
    for i,(order_idx, dbg_ilm, dbg_rpe) in enumerate(results_extra_layers):
        prior_layer = golden_set_layers[i]
        score = np.mean(np.abs(dbg_rpe.rpe_smooth-prior_layer))
        scores.append(score)
        print(score)
        print([np.mean(np.abs(e[-1].rpe_smooth-prior_layer)) for e in results_extra_layers])
    return scores

# -----------------------------
#  Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--current-bscan', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--mini-root', type=Path, required=True)
    ap.add_argument('--k', type=int, default=5, help='number of additional sample slices')
    ap.add_argument('--downsample', type=float, default=1.5)
    ap.add_argument('--max-workers', type=int, default=8)
    ap.add_argument('--debug', action='store_true', help='run serially for pdb-friendly debugging')  # <-- ADD
    ap.add_argument('--run_extra_slices', action='store_true', help='run serially for pdb-friendly debugging')  # <-- ADD
    ap.add_argument('--img_src_path')  # <-- ADD
    ap.add_argument('--z_index')  # <-- ADD
    args = ap.parse_args()

    # Load current (first) slice — always page 1
    print(f"img vol lives at path: {args.img_src_path}")
    current = np.load(args.current_bscan)
    onh_info_path = image_to_annotation_path(Path(args.img_src_path))
    try:
        current_onh_info = grab_current_onh_info(onh_info_path,args.z_index)
        print(f"onh_info vol lives at path: {onh_info_path}")
    except:
        print(f"unable to load ONH info at {onh_info_path}")
        current_onh_info = None
    current = (current,current_onh_info)

    # Load ~k additional sample slices from mini volumes deterministically
    # samples = load_mini_slices(args.mini_root, total_k=0)

    # Build processing queue: first item is the current slice (index 0), then samples (1..k)
    # work = [(0, current)] + [(i + 1, s) for i, s in enumerate(samples)]
    if args.run_extra_slices:
        import yaml
        with open("/Users/matthewhunt/Research/Iowa_Research/Han_AIR/code_files/segmentation_scratch/golden_set.yaml", "r") as f:
            golden_dict = yaml.safe_load(f)
        samples,ONH_infos,golden_set_layers = load_extra_slices_and_prior_layers(golden_dict,
                                           Path(args.mini_root))

        # golden_set_layers = load_golden_set_layers(golden_dict)
        # print(f"also processing {len(golden_set_layers)} other samples")
        # samples,ONH_infos = load_extra_slices(args.mini_root, total_k=args.k,golden_set_dict=golden_dict)
        # print(f"samples are {samples}")

        samples = list(zip(samples,ONH_infos))
        work = [(0, current[0], current[1])] + [(i + 1, s, onh) for i, (s, onh) in enumerate(samples)]
    else:
        print("skipping the following other samples")
        work = [(0, current[0],current[1])]
        golden_set_layers = None



    if args.debug or args.max_workers <= 1:
        # Serial, pdb-friendly path
        results = [sp.process_bscan_12_5_25(t) for t in work]
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.max_workers) as exe:
            futures = [exe.submit(sp.process_bscan_12_5_25, t) for t in work]
            results = [f.result() for f in futures]

    # Sort by the artificial order index (so page 1 is the current slice)
    results.sort(key=lambda x: x[0])

    
    if golden_set_layers:
        diffs_from_original = golden_set_differences(results[1:],golden_set_layers)
        print(f"diffs_from_original={diffs_from_original}")

    # Write the multipage PDF
    args.out.parent.mkdir(parents=True, exist_ok=True)


    relevant_results = []
    with PdfPages(args.out) as pdf:
        for i,(order_idx, dbg_ilm, dbg_rpe) in enumerate(results):
            # START HERE: 
            # We must calculate the difference score between our best golden sample slices, and our new labels calculated at those positions. 
            # Print the overall score, and then plot probably in sorted order of worst to best. 

            # title = 'Current slice' if order_idx == 0 else f'Sample slice #{order_idx}'
            title = 'Current slice' if order_idx == 0 else f'Sample slice MAD from original = {diffs_from_original}'
            original_line = None if i==0 else golden_set_layers[i-1] 
            render_page(pdf, title, dbg_ilm, dbg_rpe,original_line)
            relevant_results.append([order_idx,dbg_ilm.ilm_smooth,dbg_rpe.enh])
    
    # import pickle
    # print("going to pickle the entire results")
    # with open(Path(pickle_out_dir) / f"full_results", "wb") as f:
    #     pickle.dump(relevant_results,open(Path(pickle_out_dir)/f"full_results",'wb'))
    # print("done pickling")


    # Clean any sweep PDFs produced elsewhere
    try:
        spu.clean_sweep_pdfs()
    except Exception:
        pass


if __name__ == '__main__':
    main()