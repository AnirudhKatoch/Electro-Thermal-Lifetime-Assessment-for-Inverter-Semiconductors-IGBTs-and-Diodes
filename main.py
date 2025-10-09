'''
from Input_parameters_file import Input_parameters_class
from main_1 import main_1
from main_2 import main_2
import pandas as pd
import numpy as np


Loadprofile_name = "synPRO_el_family_main_1"

df = pd.read_parquet(f"Load_profiles/{Loadprofile_name}.parquet", engine="pyarrow")
P = np.array(df["P_el"])
P = P[:86400]
del df
pf = np.full(len(P), 1, dtype=float)

main_1(P=P,pf=pf,Loadprofile_name=Loadprofile_name)

print("###############################################################################################################")

Loadprofile_name = "synPRO_el_family_main_2_1_sec"
df = pd.read_parquet(f"Load_profiles/{Loadprofile_name}.parquet", engine="pyarrow")
P = np.array(df["P_el"])
del df
P = P[:86400]
pf = np.full(len(P), 1, dtype=float)

main_2(P=P,pf=pf,Loadprofile_name=Loadprofile_name)

print("###############################################################################################################")

Loadprofile_name = "synPRO_el_family_main_2_15_min"
df = pd.read_parquet(f"Load_profiles/{Loadprofile_name}.parquet", engine="pyarrow")
P = np.array(df["P_el"])
del df
P = P[:86400]
pf = np.full(len(P), 1, dtype=float)

main_2(P=P,pf=pf,Loadprofile_name=Loadprofile_name)

print("###############################################################################################################")

'''

import subprocess
import sys

def run_once(entry_script, profile):
    """
    Run a given entry script as a completely new Python process.
    When the process exits, all its memory is freed.
    """
    print(f"\n🚀 Running {entry_script} for profile {profile}")
    cmd = [sys.executable, "-u", entry_script, "--profile", profile]
    subprocess.run(cmd, check=True)
    print("✅ Completed.\n")

if __name__ == "__main__":
    # Run each simulation sequentially
    run_once("main_1_entry.py", "synPRO_el_family_main_1")
    run_once("main_2_entry.py", "synPRO_el_family_main_2_1_sec")
    run_once("main_2_entry.py", "synPRO_el_family_main_2_15_min")
