#!/usr/bin/env python3
import os
import argparse
import csv
import sys

def anonymize(root_dir, output_csv):
    # Collect all category subdirectories
    categories = sorted([
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ])

    integer_id = 1
    rows = []

    for category in categories:
        cat_path = os.path.join(root_dir, category)

        # Collect patient directories in this category
        patient_dirs = sorted([
            d for d in os.listdir(cat_path)
            if os.path.isdir(os.path.join(cat_path, d))
        ])

        for patient in patient_dirs:
            # Expect format "P{patient_id} {date}"
            if " " not in patient:
                print(f"Skipping unexpected folder name: {patient}", file=sys.stderr)
                continue

            id_part, _, date = patient.partition(" ")
            if not id_part.startswith("P"):
                print(f"Skipping non-P-prefixed folder: {patient}", file=sys.stderr)
                continue

            mrn = id_part.lstrip("P")  # strip leading "P" to get numeric MRN
            new_dir_name = f"{integer_id} {date}"

            old_dir = os.path.join(cat_path, patient)
            new_dir = os.path.join(cat_path, new_dir_name)

            # 1) Rename the directory
            # print(f"Will rename {old_dir} into {new_dir}")
            os.rename(old_dir, new_dir)

            # 2) Inside that dir, rename any files starting with the old prefix
            for fname in os.listdir(new_dir):
                if not fname.startswith(id_part):
                    continue
                rest = fname[len(id_part):]
                new_fname = f"{integer_id}{rest}"
                # print(f"would rename {fname} to {new_fname}")
                os.rename(
                    os.path.join(new_dir, fname),
                    os.path.join(new_dir, new_fname)
                )

            # 3) Record the mapping row
            rows.append([integer_id, mrn, date, category])
            integer_id += 1

    # Write out the CSV
    with open(output_csv, "w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["integer_id", "MRN", "date", "category"])
        writer.writerows(rows)

    print(f"Done! Processed {integer_id-1} patients. Mapping written to {output_csv}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Anonymize P{ID} → integer IDs and emit CSV mapping"
    )
    p.add_argument(
        "--root_dir",
        help="Path to 'all_data' directory containing category subfolders"
    )
    p.add_argument(
        "--output_csv",
        help="Path to write the mapping CSV (e.g. mapping.csv)"
    )
    args = p.parse_args()

    anonymize(args.root_dir, args.output_csv)
