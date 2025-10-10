import argparse
import pandas as pd
import numpy as np
from mother_function import main_2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    args = parser.parse_args()

    df = pd.read_parquet(f"Load_profiles/{args.profile}.parquet", engine="pyarrow")
    P = df["P"].to_numpy(dtype=np.float64)
    P = P[:86400]
    pf = np.full(len(P), 1, dtype=float)
    Q  = np.full(len(P), 10, dtype=float)
    Loadprofile_name = args.profile

    del df

    main_2(P=P, pf=pf, Q=Q,Loadprofile_name=Loadprofile_name)
