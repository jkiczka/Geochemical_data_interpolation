from data_pipeline import csv_to_xarray, download_csv
import xarray as xr

from utils import print_nan_stats

TARGET = 'nitrate_00' # for now
DROP_NA = True

if __name__ == "__main__":
    t = csv_to_xarray(download_csv("temperature", 0))
    s = csv_to_xarray(download_csv("salinity", 0))
    o = csv_to_xarray(download_csv("oxygen", 0))
    n = csv_to_xarray(download_csv("nitrate", 0))
    p = csv_to_xarray(download_csv("phosphate", 0))

    ds = xr.merge([t, s, o, n, p])
    df = ds.to_dataframe().reset_index()
    if DROP_NA:
        df.dropna(subset=[TARGET], inplace=True)
    df.to_parquet(f"data/processed/{TARGET}_train_data.parquet")