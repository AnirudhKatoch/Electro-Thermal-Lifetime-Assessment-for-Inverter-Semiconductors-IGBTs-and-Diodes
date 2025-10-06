import os
from Calculation_functions_file import Calculation_functions_class

Calculation_functions_class = Calculation_functions_class()

def save_dataframes(df_1, df_2, df_3, df_4, Location_dataframes, timestamp):

    Location_dataframes = f"{Location_dataframes}/{timestamp}"
    os.makedirs(Location_dataframes, exist_ok=True)

    #df_1.to_parquet(f"{Location_dataframes}/df_1.parquet", engine="pyarrow", compression="zstd",compression_level=7,use_dictionary=True,)
    #df_2.to_parquet(f"{Location_dataframes}/df_2.parquet", engine="pyarrow", compression="zstd",compression_level=7,use_dictionary=True,)
    #df_3.to_parquet(f"{Location_dataframes}/df_3.parquet", engine="pyarrow", compression="zstd",compression_level=7,use_dictionary=True,)
    #df_4.to_parquet(f"{Location_dataframes}/df_4.parquet", engine="pyarrow", compression="zstd",compression_level=7,use_dictionary=True,)

    Calculation_functions_class.write_dataframe_fast(df_1, f"{Location_dataframes}/df_1.parquet")
    Calculation_functions_class.write_dataframe_fast(df_2, f"{Location_dataframes}/df_2.parquet")
    Calculation_functions_class.write_dataframe_fast(df_3, f"{Location_dataframes}/df_3.parquet")
    Calculation_functions_class.write_dataframe_fast(df_4, f"{Location_dataframes}/df_4.parquet")