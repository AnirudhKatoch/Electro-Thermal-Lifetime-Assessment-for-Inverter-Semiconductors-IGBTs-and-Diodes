import argparse
import pandas as pd
import numpy as np
from main_2 import main_2  # import your actual simulation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    args = parser.parse_args()

    df = pd.read_parquet(f"Load_profiles/{args.profile}.parquet", engine="pyarrow")
    P = df["P_el"].to_numpy(dtype=np.float64)[:86400]
    pf = np.full(P.shape[0], 1.0, dtype=np.float64)
    Q = np.full(P.shape[0], 1.0, dtype=np.float64)

    main_2(P=P, pf=pf, Q=Q,Loadprofile_name=args.profile)
