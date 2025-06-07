import xarray as xr
import numpy as np

def print_nan_stats(obj: xr.DataArray | xr.Dataset, depth=None):
    if isinstance(obj, xr.Dataset):
        var_name = list(obj.data_vars)[0]
        da = obj[var_name]
    else:
        da = obj
        var_name = da.name or "unnamed"

    total = np.prod(da.shape)
    missing = np.isnan(da.values).sum()
    present = total - missing
    coverage = 100 * present / total

    print(f"Dane: {var_name}")
    print(f"  Wymiary      : {da.shape}  ({', '.join(da.dims)})")
    print(f"  Wartości     : {total:,}")
    print(f"  Braki (NaN)  : {missing:,}")
    print(f"  Pokrycie     : {coverage:.2f}%")

    if "depth" in da.dims and depth:
        shallow_da = da.sel(depth=da.depth <= depth)
        total_s = np.prod(shallow_da.shape)
        missing_s = np.isnan(shallow_da.values).sum()
        present_s = total_s - missing_s
        coverage_s = 100 * present_s / total_s

        print(f"\n📊 Pokrycie dla warstw płytkich (depth ≤ {depth} m):")
        print(f"  Głębokości : {list(shallow_da.depth.values)}")
        print(f"  Wymiary    : {shallow_da.shape}")
        print(f"  Wartości   : {total_s:,}")
        print(f"  Braki (NaN): {missing_s:,}")
        print(f"  Pokrycie   : {coverage_s:.2f}%")


