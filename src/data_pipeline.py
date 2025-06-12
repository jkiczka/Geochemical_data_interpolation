from pathlib import Path
import pandas as pd
import xarray as xr
import numpy as np
import requests, gzip, io
from scipy.interpolate import griddata
from scipy.ndimage import uniform_filter
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm

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

def inpaint_nan_2d(arr, method='nearest'):
    """Inpaint 2D NaNs using scipy griddata."""
    x, y = np.meshgrid(np.arange(arr.shape[1]), np.arange(arr.shape[0]))
    mask = ~np.isnan(arr)
    coords = np.array([x[mask], y[mask]]).T
    values = arr[mask]
    arr_filled = arr.copy()
    arr_filled[~mask] = griddata(coords, values, np.array([x[~mask], y[~mask]]).T, method=method)
    return arr_filled

def smooth_field_2d(arr, size=3):
    return uniform_filter(arr, size=size, mode='nearest')

def regrid_to_woa(source_lat, source_lon, source_data, target_lat, target_lon):
    interpolator = RegularGridInterpolator(
        (source_lat, source_lon), source_data, bounds_error=False, fill_value=np.nan
    )
    target_points = np.array([
        [lat, lon] for lat in target_lat for lon in target_lon
    ])
    result = interpolator(target_points).reshape(len(target_lat), len(target_lon))
    return result

def interpolate_to_grid(var_data, orig_lat, orig_lon, new_lat, new_lon):
    lon2d, lat2d = np.meshgrid(orig_lon, orig_lat)
    flat_points = np.column_stack((lat2d.ravel(), lon2d.ravel()))
    flat_values = var_data.ravel()

    new_lon2d, new_lat2d = np.meshgrid(new_lon, new_lat)
    interp_points = np.column_stack((new_lat2d.ravel(), new_lon2d.ravel()))

    interpolated = griddata(flat_points, flat_values, interp_points, method='linear')
    return interpolated.reshape(len(new_lat), len(new_lon))

def process_woa_grid(verbose=True):
    if verbose:
        print("Loading .parquet file...")

    df_nitrate = pd.read_parquet("data/processed/nitrate_00_train_data.parquet")
    
    ds = df_nitrate.set_index(["depth", "lat", "lon"]).to_xarray()

    if verbose:
        print("Inputting NaNs...")

    for var in ['temperature_00', 'salinity_00', 'oxygen_00', 'nitrate_00', 'phosphate_00']:
        for d in ds.depth.values:
            arr = ds[var].sel(depth=d).values
            ds[var].loc[dict(depth=d)] = inpaint_nan_2d(arr)

    if verbose:
        print("Smoothing 2D field...")

    for var in ['temperature_00', 'salinity_00', 'oxygen_00', 'nitrate_00', 'phosphate_00']:
        for d in ds.depth.values:
            arr = ds[var].sel(depth=d).values
            ds[var].loc[dict(depth=d)] = smooth_field_2d(arr)

    # Load WOA lat/lon from file or define them
    woa_lat = np.arange(-90, 90.25, 0.25)
    woa_lon = np.arange(-180, 180.25, 0.25)

    if verbose:
        print("Interpolating values...")

    new_ds = {}
    for var in ['temperature_00', 'salinity_00', 'oxygen_00', 'nitrate_00', 'phosphate_00']:
        var_interp = []
        for d in tqdm(ds.depth.values, desc=f'Interpolating {var}', disable=not verbose):
            orig_arr = ds[var].sel(depth=d).values
            interp_arr = interpolate_to_grid(orig_arr, ds.lat.values, ds.lon.values, woa_lat, woa_lon)
            var_interp.append(interp_arr)
        new_ds[var] = (("depth", "lat", "lon"), np.array(var_interp))

    # Build the new Dataset on the WOA grid
    ds_woa = xr.Dataset(
        data_vars=new_ds,
        coords={
            "depth": ds.depth.values,
            "lat": woa_lat,
            "lon": woa_lon
        }
    )

    if verbose:
        print("Converting to Dataframe...")

    df_woa = ds_woa.to_dataframe().reset_index()

    print("Done!")

    return df_woa, ds_woa

def get_woa_grid_shallow(ds_woa):
    df_woa_shallow = ds_woa.isel(depth=0).to_dataframe().reset_index()
    return df_woa_shallow

def preprocess_yang_dataset():
    df_yang = pd.read_csv("data/raw/surface_n2o_compilation.csv")

    # rename columns for clarity
    df_yang.rename(columns={
        'latitude': 'lat',
        'longitude': 'lon',
    }, inplace=True)

    YANG_COLUMNS = ['lat', 'lon', 'depth', 'n2o_ppb', 'n2o_nM', 'dn2o_ppb', 'atmPressure', 'salinity']

    # Select only the relevant columns
    df_yang = df_yang[YANG_COLUMNS]

    df_yang = df_yang.dropna(subset=['lat', 'lon'])
    df_yang['lon'] = df_yang['lon'].apply(lambda x: x - 360 if x > 180 else x)

    df_yang.groupby(['lat', 'lon', 'depth']).mean().reset_index(inplace=True)

    return df_yang

def find_closest_points_on_grid(df_yang, df_woa_shallow):
     # Round yang coordinates to 0.25° resolution
    df_yang['lat_round'] = np.round(df_yang['lat'] * 4) / 4
    df_yang['lon_round'] = np.round(df_yang['lon'] * 4) / 4

    df_yang.index.name = 'yang_index'

    df_joined = pd.merge(
          df_yang,
          df_woa_shallow,
          how='left',
          left_on=['lat_round', 'lon_round'],
          right_on=['lat', 'lon'],
          suffixes=('_yang', '_woa')
    )

    df_joined = df_joined.drop(columns=['lat_round', 'lon_round'])

    return df_joined

def adjust_yang_grid_shallow(df_yang, df_woa_shallow):
    df_yang_shallow = df_yang.loc[df_yang.groupby(['lat', 'lon'])['depth'].idxmin()]

    df_joined = find_closest_points_on_grid(df_yang_shallow, df_woa_shallow)

    return df_joined


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
