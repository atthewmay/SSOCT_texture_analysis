import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))  # adds Han_AIR/ to path
import file_utils
import os
import shutil

def copy_volumes(fp):
    if not "cube_z" in str(fp):
        return
    destination_path = '/Users/matthewhunt/Research/Iowa_Research/Han_AIR/data_all_volumes'
    f_name = os.path.basename(fp)
    # print(f"will copy from {fp} to {os.path.join(destination_path,f_name)}")
    shutil.copy(fp,os.path.join(destination_path,f_name))
    return


file_utils.crawl_dirs("data",copy_volumes)
