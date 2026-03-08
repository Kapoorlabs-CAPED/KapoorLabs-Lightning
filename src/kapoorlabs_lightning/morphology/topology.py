"""
Persistent homology computation for spatial point configurations.

Computes Vietoris-Rips persistent homology on cell centroid data
to capture topological features (connected components H0, loops H1,
voids H2) across timepoints.

Requires: ripser, persim
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, List

from scipy.spatial.distance import pdist, squareform

try:
    from ripser import ripser
except ImportError:
    ripser = None

try:
    from persim import plot_diagrams
except ImportError:
    plot_diagrams = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    plt = None
    sns = None


def vietoris_rips_at_t(
    df: pd.DataFrame,
    t,
    spatial_cols: Tuple[str, ...] = ("z", "y", "x"),
    max_dim: int = 2,
    max_edge: Optional[float] = None,
    metric: str = "euclidean",
    normalise: bool = False,
    plot: bool = False,
    use_explicit_distance: bool = True,
) -> List[np.ndarray]:
    """
    Compute Vietoris-Rips persistent homology at a single timepoint.

    Args:
        df: DataFrame with a 't' column and coordinate columns.
        t: Timepoint to compute homology for.
        spatial_cols: Column names for spatial coordinates.
        max_dim: Maximum homological dimension.
        max_edge: Maximum edge length threshold.
        metric: Distance metric for pairwise distances.
        normalise: Whether to z-score normalize coordinates.
        plot: If True, display persistence diagram.
        use_explicit_distance: If True, pre-compute distance matrix.

    Returns:
        List of persistence diagram arrays, one per dimension.
        diagrams[dim] has shape (n_features, 2) with [birth, death].
    """
    if ripser is None:
        raise ImportError(
            "ripser is required for topology computation. "
            "Install with: pip install ripser"
        )

    pts = (
        df.loc[df["t"] == t, spatial_cols]
        .dropna()
        .astype(float)
        .to_numpy(dtype=np.float64)
    )

    if pts.size == 0:
        raise ValueError(f"No valid points at t = {t}")

    if normalise:
        std = pts.std(0, ddof=0)
        std[std == 0] = 1.0
        pts = (pts - pts.mean(0)) / std

    if use_explicit_distance:
        dists = squareform(pdist(pts, metric=metric)).astype(np.float64)
        if np.isnan(dists).any():
            raise ValueError(
                f"Distance matrix for t={t} contains NaN values"
            )
        dgms = ripser(
            dists,
            distance_matrix=True,
            maxdim=max_dim,
            thresh=max_edge if max_edge is not None else np.inf,
        )["dgms"]
    else:
        dgms = ripser(
            pts,
            maxdim=max_dim,
            thresh=max_edge,
            metric=metric,
        )["dgms"]

    if plot and plot_diagrams is not None:
        plot_diagrams(dgms, show=True, title=f"t = {t}")

    return dgms


def diagrams_over_time(
    df: pd.DataFrame,
    time_col: str = "t",
    **vr_kwargs,
) -> Dict:
    """
    Compute persistent homology diagrams for all timepoints.

    Args:
        df: DataFrame with time and coordinate columns.
        time_col: Name of the time column.
        **vr_kwargs: Additional arguments passed to vietoris_rips_at_t.

    Returns:
        Dict mapping timepoint → list of persistence diagram arrays.
    """
    from tqdm import tqdm

    unique_times = np.sort(df[time_col].unique())
    diags = {}
    for t in tqdm(unique_times, desc="VR per frame"):
        diags[t] = vietoris_rips_at_t(df, t, plot=False, **vr_kwargs)
    return diags


def save_barcodes_and_stats(
    diagrams_by_time: Dict,
    dims: Tuple[int, ...] = (0, 1),
    save_dir: str = "barcodes_per_frame",
    max_bars: Optional[int] = None,
    plot_joint_hist: bool = True,
    csv_loop_stats: bool = True,
):
    """
    Save barcode plots and persistence statistics for each timepoint.

    For every timepoint saves:
    - Barcode plot PNG (H_dim for each dim in dims)
    - Optional: combined histogram/KDE of H1 persistence across time
    - Optional: CSV per timepoint with birth/death/persistence (H1)

    Args:
        diagrams_by_time: Dict from diagrams_over_time().
        dims: Homological dimensions to plot.
        save_dir: Output directory.
        max_bars: Maximum number of bars to display per barcode.
        plot_joint_hist: Whether to plot combined H1 persistence KDE.
        csv_loop_stats: Whether to export per-timepoint CSV files.
    """
    if plt is None:
        raise ImportError("matplotlib is required for barcode visualization")

    save_dir = Path(save_dir)
    png_dir = save_dir / "png"
    csv_dir = save_dir / "csv"
    hist_path = save_dir / "combined_histogram_H1_persistence.png"

    png_dir.mkdir(parents=True, exist_ok=True)
    if csv_loop_stats:
        csv_dir.mkdir(exist_ok=True)

    all_persistence = []

    for t, dgms in diagrams_by_time.items():
        fig, axs = plt.subplots(
            len(dims), 1, figsize=(8, 2.5 * len(dims)), squeeze=False
        )
        for i, dim in enumerate(dims):
            ax = axs[i, 0]
            if dim >= len(dgms) or dgms[dim].size == 0:
                ax.set_axis_off()
                continue

            diag = dgms[dim]
            if max_bars is not None and len(diag) > max_bars:
                pers = diag[:, 1] - diag[:, 0]
                idx = np.argsort(-pers)[:max_bars]
                diag = diag[idx]

            for j, (b, d) in enumerate(diag):
                ax.hlines(j, b, d, color="tab:blue")
                ax.plot([b, d], [j, j], "k.", ms=2)
            ax.set_title(f"H{dim} barcode at t={t}")
            ax.set_xlabel("Filtration scale")
            ax.set_ylabel("Feature index")
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(png_dir / f"barcode_t{int(t):04d}.png", dpi=200)
        plt.close(fig)

        if plot_joint_hist and len(dgms) > 1 and dgms[1].size:
            persistence = dgms[1][:, 1] - dgms[1][:, 0]
            all_persistence.append(
                pd.DataFrame({"persistence": persistence, "time": t})
            )

        if csv_loop_stats and len(dgms) > 1 and dgms[1].size:
            df_out = pd.DataFrame(dgms[1], columns=["birth", "death"])
            df_out["persistence"] = df_out["death"] - df_out["birth"]
            df_out.to_csv(
                csv_dir / f"loops_t{int(t):04d}.csv", index=False
            )

    if plot_joint_hist and all_persistence and sns is not None:
        full_df = pd.concat(all_persistence, ignore_index=True)
        full_df["bin"] = (full_df["time"] // 50).astype(int)
        n_bins = full_df["bin"].nunique()
        palette = sns.color_palette("tab10", n_bins)

        plt.figure(figsize=(10, 6))
        for bin_id, colour in zip(
            sorted(full_df["bin"].unique()), palette
        ):
            times_in_bin = full_df.loc[
                full_df["bin"] == bin_id, "time"
            ].unique()
            for t in times_in_bin:
                subset = full_df[full_df["time"] == t]
                label = (
                    f"{bin_id*50}-{bin_id*50+49}"
                    if t == times_in_bin[0]
                    else None
                )
                sns.kdeplot(
                    subset["persistence"],
                    color=colour,
                    label=label,
                    linewidth=1.4,
                    alpha=0.40,
                )

        plt.title("H1 persistence KDEs (grouped in 50-frame bins)")
        plt.xlabel("Persistence (death - birth)")
        plt.ylabel("Density")
        plt.legend(title="Time bin (t)")
        plt.tight_layout()
        plt.savefig(hist_path, dpi=300)
        plt.close()
