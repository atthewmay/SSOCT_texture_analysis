#!/usr/bin/env python3
"""
Thin single-volume demo wrapper around code_files/setup_data/02_segment_ILM_RPE.py.

This intentionally does not duplicate the segmentation/save logic.  The canonical
implementation remains in 02_segment_ILM_RPE.py:
    - process_volume_lite(...)
    - extract_lite(...)
    - collate_stackable(...)
    - per-z and stacked NPZ saving
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


# --- project path ---
# This file is intended to live beside 02_segment_ILM_RPE.py in code_files/setup_data/.
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
CODE_FILES_DIR = THIS_FILE.parents[1]

sys.path.append(str(CODE_FILES_DIR))
sys.path.append(str(REPO_ROOT))

import code_files.segmentation_code.segmentation_pipelines as sp


def _load_02_segment_module():
    """
    Load 02_segment_ILM_RPE.py despite the leading-number filename.

    Normal Python imports cannot do:
        import 02_segment_ILM_RPE

    so use importlib and keep the source of truth in that script.
    """
    script_path = THIS_FILE.with_name("02_segment_ILM_RPE.py")
    if not script_path.exists():
        raise FileNotFoundError(f"Could not find expected script: {script_path}")

    spec = importlib.util.spec_from_file_location("segment_ilm_rpe_02", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module from {script_path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _get_pipeline_attr(name: str):
    """
    Resolve a pipeline name from segmentation_pipelines.py.

    Accepts either:
      - exact names, e.g. RPE_STEPS_unified_3_19_26
      - shortcuts below, e.g. latest
    """
    shortcuts = {
        "latest": "RPE_STEPS_unified_3_19_26",
    }
    attr = shortcuts.get(name, name)

    if not hasattr(sp, attr):
        available = sorted(
            k for k in dir(sp)
            if k.startswith(("RPE_STEPS", "ILM_STEPS"))
        )
        raise ValueError(
            f"Unknown pipeline '{name}' -> '{attr}'.\n"
            f"Available pipeline attrs include:\n  " + "\n  ".join(available)
        )

    return getattr(sp, attr), attr


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Run one OCT volume through the layer segmentation code using the "
            "same implementation as 02_segment_ILM_RPE.py."
        )
    )

    p.add_argument(
        "--volume",
        required=True,
        type=Path,
        help="Full path to a single .img volume.",
    )
    p.add_argument(
        "--outputs_root",
        required=True,
        type=Path,
        help="Root output directory. A subdirectory named after the volume stem is created.",
    )
    p.add_argument(
        "--annotation_root",
        required=True,
        type=Path,
        help="Root directory containing ONH/annotation zarrs used by fu.load_vol_and_annotation.",
    )
    p.add_argument(
        "--z_step",
        "--z_stride",
        dest="z_step",
        type=int,
        default=250,
        help="Segment every Nth B-scan. Use 1 for all slices.",
    )
    p.add_argument(
        "--max_workers",
        type=int,
        default=1,
        help="Number of worker processes. Use 1 for easiest debugging.",
    )
    p.add_argument(
        "--pipeline",
        default="latest",
        help=(
            "RPE pipeline shortcut or exact attr from segmentation_pipelines.py. "
            "Default: latest -> RPE_STEPS_unified_3_19_26."
        ),
    )
    p.add_argument(
        "--ilm_pipeline",
        default="ILM_STEPS_2_28",
        help=(
            "ILM pipeline shortcut or exact attr from segmentation_pipelines.py. "
            "Default: ILM_STEPS_2_28."
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()

    seg02 = _load_02_segment_module()

    rpe_steps, rpe_name = _get_pipeline_attr(args.pipeline)
    ilm_steps, ilm_name = _get_pipeline_attr(args.ilm_pipeline)

    print(f"Processing one volume: {args.volume}")
    print(f"RPE pipeline: {rpe_name}")
    print(f"ILM pipeline: {ilm_name}")
    print(f"z_step: {args.z_step}")
    print(f"outputs_root: {args.outputs_root}")

    out = seg02.process_volume_lite(
        args.volume,
        z_step=args.z_step,
        max_workers=args.max_workers,
        rpe_steps=rpe_steps,
        ilm_steps=ilm_steps,
        out_dir=args.outputs_root,
        annotation_root=args.annotation_root,
    )

    print(f"saved -> {out}")
    print("DONE")
    return out


if __name__ == "__main__":
    main()


