import os


def save_dataframes(df_1, df_2, df_3, df_4, Location_dataframes, timestamp):

    Location_dataframes = f"{Location_dataframes}/{timestamp}"
    os.makedirs(Location_dataframes, exist_ok=True)

    df_1.to_parquet(f"{Location_dataframes}/df_1.parquet", engine="pyarrow", compression="zstd")
    df_2.to_parquet(f"{Location_dataframes}/df_2.parquet", engine="pyarrow", compression="zstd")
    df_3.to_parquet(f"{Location_dataframes}/df_3.parquet", engine="pyarrow", compression="zstd")
    df_4.to_parquet(f"{Location_dataframes}/df_4.parquet", engine="pyarrow", compression="zstd")