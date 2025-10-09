import argparse
import pandas as pd
import numpy as np
from main_2 import main_2  # import your actual simulation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    args = parser.parse_args()

    df = pd.read_parquet(f"Load_profiles/{args.profile}.parquet", engine="pyarrow")
    P = df["P"].to_numpy(dtype=np.float64)
    pf = df["pf"].to_numpy(dtype=np.float64)
    Q = df["Q"].to_numpy(dtype=np.float64)
    Loadprofile_name = args.profile

    main_2(P=P, pf=pf, Q=Q,Loadprofile_name=Loadprofile_name)
