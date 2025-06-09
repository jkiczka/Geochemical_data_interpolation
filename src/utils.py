import pandas as pd
from typing import Sequence
from matplotlib import pyplot as plt
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


def plot_true_vs_pred(
    test_df: pd.DataFrame,
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
    *,
    target: str = "target",
    cmap: str = "viridis",
    point_size: int = 5,
) -> None:
    """Scatter side‑by‑side maps of *ground truth* and *predictions*.

    Parameters
    ----------
    test_df : pd.DataFrame
        Must contain `lat` and `lon` columns.
    y_true, y_pred : array‑like
        True and predicted values, aligned with `test_df` rows.
    target : str, optional
        Variable name for titles/color bars.
    cmap : str, optional
        Matplotlib colour‑map.
    point_size : int, optional
        Marker size for scatter.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    # ── ground truth ──
    sc0 = axes[0].scatter(
        test_df["lon"],
        test_df["lat"],
        c=y_true,
        cmap=cmap,
        s=point_size,
    )
    plt.colorbar(sc0, ax=axes[0], label=f"True {target}")
    axes[0].set(
        title=f"Ground truth – {target}",
        xlabel="Longitude",
        ylabel="Latitude",
    )
    axes[0].grid(ls="--", alpha=0.3)

    # ── predictions ──
    sc1 = axes[1].scatter(
        test_df["lon"],
        test_df["lat"],
        c=y_pred,
        cmap=cmap,
        s=point_size,
    )
    plt.colorbar(sc1, ax=axes[1], label=f"Predicted {target}")
    axes[1].set(
        title=f"RandomForest – {target}",
        xlabel="Longitude",
        ylabel="Latitude",
    )
    axes[1].grid(ls="--", alpha=0.3)

    plt.show()
