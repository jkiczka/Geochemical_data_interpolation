from pathlib import Path
import pandas as pd
import xarray as xr
import numpy as np
import requests, gzip, io

DEPTHS = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,
        100,125,150,175,200,225,250,275,300,325,350,375,400,425,450,
        475,500,550,600,650,700,750,800,850,900,950,1000,1050,1100,
        1150,1200,1250,1300,1350,1400,1450,1500]

BASE_URL = "https://www.ncei.noaa.gov/data/oceans/woa/WOA23/DATA/"

_TPL = {
    "temperature": "temperature/csv/B5C2/1.00/woa23_B5C2_t{m}mn01.csv.gz",
    "salinity":    "salinity/csv/B5C2/1.00/woa23_B5C2_s{m}mn01.csv.gz",
    "oxygen":      "oxygen/csv/all/1.00/woa23_all_o{m}mn01.csv.gz",
    "nitrate":     "nitrate/csv/all/1.00/woa23_all_n{m}mn01.csv.gz",
    "phosphate":   "phosphate/csv/all/1.00/woa23_all_p{m}mn01.csv.gz",
}

def download_csv(
    variable: str,
    month: int,
    outdir: Path = Path("data/raw"),
) -> Path:
    var = variable.lower()
    if var not in _TPL:
        raise ValueError(f"unsupported variable '{variable}'")

    url = BASE_URL + _TPL[var].format(m=f"{month:02d}")
    outdir.mkdir(parents=True, exist_ok=True)
    dest = outdir / f"{var}_{month:02d}.csv"

    if dest.exists():
        return dest

    r = requests.get(url, timeout=60)
    r.raise_for_status()
    csv_bytes = gzip.decompress(r.content)
    dest.write_bytes(csv_bytes)

    return dest

def csv_to_xarray(path: Path) -> xr.Dataset:
    import csv

    rows = []
    with open(path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if row and not row[0].startswith("#"):
                rows.append(row)

    data = []
    for row in rows:
        values = [float(x) if x else np.nan for x in row]
        while len(values) < 2 + len(DEPTHS):
            values.append(np.nan)
        values = values[:2 + len(DEPTHS)]
        data.append(values)

    df = pd.DataFrame(data, columns=["lat", "lon", *DEPTHS])
    
    ds = (
        df.set_index(["lat", "lon"])
            .to_xarray()
            .to_array(name=Path(path).stem)
            .transpose("lat", "lon", "variable")
            .rename(variable="depth")
            .assign_coords(depth=DEPTHS)
    )
    return ds.squeeze()

def random_test_patches_mask(df: pd.DataFrame, grid_size=(20, 20), n_patches=10, seed=42) -> pd.Series:
    """
    Losuje n_patches z globalnej siatki i tworzy maskę testową.
    
    Args:
        df: DataFrame z kolumnami 'lat' i 'lon'.
        grid_size: (n_lat_bins, n_lon_bins) – ile kafli w pionie i poziomie.
        n_patches: ile patchy ma trafić do testu.
        seed: dla powtarzalności.

    Returns:
        test_mask: pd.Series[bool] z True dla testowych patchy.
    """
    lat_bins = np.linspace(df["lat"].min(), df["lat"].max(), grid_size[0] + 1)
    lon_bins = np.linspace(df["lon"].min(), df["lon"].max(), grid_size[1] + 1)

    lat_inds = np.digitize(df["lat"], bins=lat_bins) - 1
    lon_inds = np.digitize(df["lon"], bins=lon_bins) - 1

    patch_ids = lat_inds * grid_size[1] + lon_inds
    df = df.copy()
    df["patch_id"] = patch_ids

    unique_patches = df["patch_id"].unique()
    rng = np.random.default_rng(seed)
    test_patch_ids = rng.choice(unique_patches, size=n_patches, replace=False)

    test_mask = df["patch_id"].isin(test_patch_ids)
    return test_mask
