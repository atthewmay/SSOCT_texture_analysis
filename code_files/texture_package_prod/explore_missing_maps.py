from collections import Counter
from pathlib import Path
import numpy as np

root = Path()

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Describe what this script does."
    )

    parser.add_argument(
        "--root",
        type=str,
        default = '/Volumes/T9/iowa_research/Han_AIR_Dec_2025/results/enface_maps/4_21_26',
        help="Path to the input file or directory.",
    )
    return parser


parser = build_parser()
args = parser.parse_args()
root = Path(args.root)


subdirs = sorted(
    d for d in root.iterdir()
    if d.is_dir() and (d / 'enface_maps.npz').exists()
)

feature_sets = {}
counter = Counter()

for subd in subdirs:
    with np.load(subd / 'enface_maps.npz') as a:
        feats = set(a.files)
    feature_sets[subd.name] = feats
    counter.update(feats)

all_features = set(counter)

print("feature occurrence counts:")
for feat, n in sorted(counter.items(), key=lambda kv: (kv[1], kv[0])):
    print(f"{n:>3}/{len(subdirs)}  {feat}")

print("\nsubdirs with missing features:")
for subd_name, feats in sorted(feature_sets.items()):
    missing = sorted(all_features - feats)
    if missing:
        print(f"\n{subd_name}")
        for feat in missing:
            print(f"  {feat}")
    if not missing:
        print(f"\n{subd_name} has all features!")
        # for feat in all_features:
            # print(f"  {feat}")