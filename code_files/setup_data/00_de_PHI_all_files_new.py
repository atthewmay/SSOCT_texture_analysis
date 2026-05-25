#!/usr/bin/env python3
import os
import argparse
import csv
import sys
from pathlib import Path
import shutil
import re
from code_files import file_utils as f

def load_master_mapping(master_csv):
    """
    Returns:
        mrn_to_id: dict[str, int]
        max_id: int
    """
    mrn_to_id = {}
    max_id = 0
    # with open(master_csv, "r", newline="") as f:
    with open(master_csv, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        # Expect at least: integer_id, MRN
        for row in reader:
            iid = int(row["integer_id"])
            mrn = row.get("MRN", "")
            mrn_to_id[mrn] = {'iid':iid,'date':row['date']}
            if iid > max_id:
                max_id = iid
    return mrn_to_id, max_id

def append_new_mappings(master_csv, new_rows):
    """
    new_rows: list[dict] with keys: integer_id, MRN, date, category
    Appends only; creates file with header if needed.
    """
    if not master_csv:
        return

    master_path = Path(master_csv)
    file_exists = master_path.exists()
    file_nonempty = file_exists and master_path.stat().st_size > 0

    master_path.parent.mkdir(parents=True, exist_ok=True)


    # If appending to an existing nonempty file, ensure it ends with newline.
    if file_nonempty:
        with open(master_path, "rb+") as f:
            f.seek(-1, 2)
            last_char = f.read(1)
            if last_char not in (b"\n", b"\r"):
                f.write(b"\n")

    with open(master_path, "a", newline="") as f:
        fieldnames = ["integer_id", "MRN", "date", "category"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            print(f"starting a new master csv at {master_path}")
            writer.writeheader()
        for r in new_rows:
            print(f"adding new rows to {master_path}")
            writer.writerow(r)

def anonymize(root_dir, output_csv, master_csv,central_image_dir):
    """you should set as the root the direcotory above the directory containing all the P{MRN}"""
    # Load persistent mapping so IDs are stable across runs
    mrn_to_id, max_id = load_master_mapping(master_csv)
    next_id = max_id + 1

    categories = sorted([
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ])

    rows_for_run = []         # what you write to output_csv (could include MRN if you want)
    new_rows_for_master = []  # only newly-created MRN→ID mappings to append

    for category in categories:
        cat_path = os.path.join(root_dir, category)
        # print(cat_path)

        patient_dirs = sorted([
            d for d in os.listdir(cat_path)
            # if os.path.isdir(os.path.join(cat_path, d))
        ])

        for patient in patient_dirs:
            # Expect format "P{MRN} {date}"
            print(patient)
            if " " not in patient:
                print(f"Skipping unexpected folder name: {patient}", file=sys.stderr)
                continue

            id_part, _, date = patient.partition(" ")
            if not id_part.startswith("P"):
                print(f"Skipping non-P-prefixed folder: {patient}", file=sys.stderr)
                continue

            mrn = id_part.lstrip("P")

            # NEW: stable integer per MRN across runs
            if mrn in mrn_to_id:
                integer_id = mrn_to_id[mrn]['iid']
                iid_date = mrn_to_id[mrn]['date']
                assert date == iid_date #only expect a single date per patient, a
                print(f"We already have mrn={mrn} with id={integer_id}, with same date = {iid_date} in the master sheet. Skipping processing of this entry")
                continue
            else:
                integer_id = next_id
                mrn_to_id[mrn] = integer_id
                next_id += 1
                new_rows_for_master.append({
                    "integer_id": integer_id,
                    "MRN": mrn,
                    "date": date,
                    "category": category,
                })

            rows_for_run.append([integer_id, mrn, date, category])

            
            # NOW WE COPY THE FILES AFTER RENAMING
            patient_path = os.path.join(cat_path,patient)
            for file in os.listdir(patient_path):
                if not ("cube_z" in str(file) and "1024x1024" in str(file) and "_Cube" in str(file)):
                    continue
                f_name = os.path.basename(file)
                fp = os.path.join(patient_path,file)
                rest_of_name = f_name.split(mrn)[-1]
                rest_of_name = f.strip_sn4(rest_of_name)
                new_f_name = f"{integer_id}{rest_of_name}"
                # print(fp)
                print(f"copying {fp} to {central_image_dir} with new name {new_f_name}")
                shutil.copy(fp,os.path.join(central_image_dir,new_f_name))


    # Write run CSV (you can keep MRN here if you want; or remove MRN if this CSV leaves the secure area)
    with open(output_csv, "w", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["integer_id", "MRN", "date", "category"])
        writer.writerows(rows_for_run)

    # Append only newly-created mappings to the persistent master mapping
    print(f"new_rows_for_master = {new_rows_for_master}")
    append_new_mappings(master_csv, new_rows_for_master)

    print(f"Done! Processed {len(rows_for_run)} folders. Run log: {output_csv}")
    if master_csv:
        print(f"Master mapping updated with {len(new_rows_for_master)} new patients: {master_csv}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Anonymize P{MRN} → stable integer IDs (persistent mapping) and emit CSV mapping"
    )
    p.add_argument("--root_dir", required=True,
                   help="Path to directory containing category subfolders")
    p.add_argument("--output_csv", required=True,
                   help="Path to write this run's mapping CSV (contains MRN unless you edit it)")
    p.add_argument("--master_csv", required=True,
                   help="Persistent MRN→integer_id master CSV (store securely; reused across runs)")
    p.add_argument("--central_image_dir", required=True,
                   help="final resting place for .img files!")
    args = p.parse_args()

    anonymize(args.root_dir, args.output_csv, args.master_csv, args.central_image_dir)
