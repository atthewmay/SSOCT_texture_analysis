#!/usr/bin/env python3
#REVIEWED
from __future__ import annotations

import argparse
import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

import code_files.file_utils as fu
import code_files.segmentation_code.segmentation_pipelines as sp

VOLUMES_ROOT = Path("/Volumes/T9/iowa_research/Han_AIR_Dec_2025/data_volumes/data_all_volumes2")
ANNOTATION_ROOT = Path("/Users/matthewhunt/Research/Iowa_Research/Han_AIR/annotations_dir/full_annotations_2_19_26/")
ILM_PIPELINE = sp.ILM_STEPS_2_28


@dataclass(frozen=True)
class SegmentationCase:
    volume_number: str
    side: str
    slice_number: int


def get_pipeline(name: str, module_file: Path | None = None):
    module = sp
    if module_file:
        spec = importlib.util.spec_from_file_location("external_pipeline", module_file)
        if spec is None or spec.loader is None:
            raise ImportError(module_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    try:
        return getattr(module, name)
    except AttributeError as e:
        raise ValueError(f"No pipeline named {name!r} in {module.__name__}") from e


def load_cases(yaml_arg: str) -> list[SegmentationCase]:
    text = yaml_arg if yaml_arg.lstrip().startswith(("-", "[", "{")) else Path(yaml_arg).read_text()
    rows = yaml.safe_load(text)
    return [
        SegmentationCase(
            str(row.get("volume_number") or row.get("vol_id")),
            str(row["side"]).upper(),
            int(row.get("slice_number") if row.get("slice_number") is not None else row["slice_id"]),
        )
        for row in rows
    ]


def find_volume(case: SegmentationCase, volumes_root: Path = VOLUMES_ROOT) -> Path:
    matches = sorted(
        p for p in volumes_root.rglob(f"{case.volume_number}_*.img")
        if fu.extract_eye_side(p.name) == case.side.upper()
    )
    if len(matches) != 1:
        raise ValueError(f"Expected one volume for {case}; found {matches}")
    return matches[0]


def process_case(
    case: SegmentationCase,
    rpe_steps,
    idx: int = 0,
    volumes_root: Path = VOLUMES_ROOT,
    annotation_root: Path = ANNOTATION_ROOT,
):
    vol_path = find_volume(case, volumes_root)
    vol, onh = fu.load_vol_and_annotation(vol_path, annotation_root)
    z = case.slice_number
    return sp.process_bscan_1_3_26(
        (idx, vol[z].copy(), onh[z, :, :][...], f"{vol_path.stem}_idx:{z}"),
        False,
        rpe_steps,
        ILM_PIPELINE,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--vol-id")
    p.add_argument("--side", choices=("OD", "OS"))
    p.add_argument("--slice-id", type=int)
    p.add_argument("--cases-yaml", help="YAML path or inline YAML list")
    p.add_argument("--pipeline", required=True)
    p.add_argument("--pipeline-module", type=Path)
    p.add_argument("--volumes-root", type=Path, default=VOLUMES_ROOT)
    p.add_argument("--annotation-root", type=Path, default=ANNOTATION_ROOT)
    args = p.parse_args()

    if args.cases_yaml:
        cases = load_cases(args.cases_yaml)
    else:
        if None in (args.vol_id, args.side, args.slice_id):
            p.error("Give --vol-id/--side/--slice-id, or --cases-yaml")
        cases = [SegmentationCase(args.vol_id, args.side, args.slice_id)]

    rpe_steps = get_pipeline(args.pipeline, args.pipeline_module)
    for idx, case in enumerate(cases):
        process_case(case, rpe_steps, idx, args.volumes_root, args.annotation_root)
        print(f"done: {case}")


if __name__ == "__main__":
    main()

"""
Single case using an external pipeline module:
Nice safe design to be used in secure docker container w/o imports being put explicity in production code. 

python run_segmentation_cases.py 
    --vol-id 2 
    --side OD 
    --slice-id 544 
    --pipeline-module /untrusted/test_pipeline.py 
    --pipeline PIPELINE


Multiple cases from a YAML file:

python run_segmentation_cases.py 
    --cases-yaml /untrusted/cases.yaml 
    --pipeline-module /untrusted/test_pipeline.py 
    --pipeline PIPELINE


Optional explicit Docker paths:

python run_segmentation_cases.py 
    --vol-id 24 
    --side OD 
    --slice-id 512 
    --pipeline-module /code_untrusted/test_pipeline.py 
    --pipeline PIPELINE 
    --volumes-root /Volumes/T9/iowa_research/Han_AIR_Dec_2025/data_volumes/data_all_volumes2
    --annotation-root /Users/matthewhunt/Research/Iowa_Research/Han_AIR/annotations_dir/full_annotations_2_19_26

python code_files/segmentation_code/run_segmentation_cases.py --vol-id 24 --side OD --slice-id 512 --pipeline-module /code_untrusted/test_pipeline.py --pipeline PIPELINE --volumes-root /Volumes/T9/iowa_research/Han_AIR_Dec_2025/data_volumes/data_all_volumes2 --annotation-root /Users/matthewhunt/Research/Iowa_Research/Han_AIR/annotations_dir/full_annotations_2_19_26


"""